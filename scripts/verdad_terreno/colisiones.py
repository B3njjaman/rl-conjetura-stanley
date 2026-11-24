"""Verdad de terreno de los Enfoques 1 y 2: cuantos pares NO isomorfos con el mismo polinomio
existen de verdad, revisando TODOS los objetos de cada tamano. No usa redes neuronales.

    python scripts/verdad_terreno/colisiones.py grafos --n 5 6 7
    python scripts/verdad_terreno/colisiones.py arboles --n 8 9 10 11 12 --k 3 4

Grafos: todos los grafos conexos de n vertices (atlas de networkx, n <= 7), agrupados por el
vector de U(x, y=0) (equivalente a la funcion simetrica cromatica X_G).
Arboles: todos los arboles no isomorfos de n vertices, agrupados por el U restringido de nivel k.
Como los objetos enumerados son representantes de clases de isomorfismo distintas, todo grupo
con 2 o mas elementos es una coleccion de objetos no isomorfos con el mismo polinomio.

Deja un JSON por caso en resultados/verdad_terreno/.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from stanley_rl.upoly import (padres_desde_arbol_nx, particiones, base_arboles,  # noqa: E402
                              u_signed_lote, uk_lote)

SALIDA = RAIZ / "resultados" / "verdad_terreno"


def agrupar(vectores):
    grupos = defaultdict(list)
    for i, v in enumerate(vectores):
        grupos[v.tobytes()].append(i)
    return [g for g in grupos.values() if len(g) > 1]


def grafos(n):
    t0 = time.time()
    G = [g for g in nx.graph_atlas_g() if g.number_of_nodes() == n and nx.is_connected(g)]
    A = np.zeros((len(G), n), dtype=np.int64)
    for s, g in enumerate(G):
        for u, v in g.edges():
            A[s, u] |= 1 << v
            A[s, v] |= 1 << u
    V = u_signed_lote(A, n)
    colis = agrupar(V)
    base = particiones(n)
    ejemplos = []
    for grupo in colis[:20]:
        ejemplos.append(dict(
            grafos=[sorted(G[i].edges()) for i in grupo],
            n_aristas=[G[i].number_of_edges() for i in grupo],
            vector={str(base[j]): int(V[grupo[0], j]) for j in np.nonzero(V[grupo[0]])[0]}))
    res = dict(n=n, grafos_conexos=len(G), grupos_con_colision=len(colis),
               grafos_en_colision=sum(len(g) for g in colis),
               pares_no_isomorfos=sum(len(g) * (len(g) - 1) // 2 for g in colis),
               min_aristas_en_colision=min((G[i].number_of_edges() for g in colis for i in g), default=None),
               segundos=round(time.time() - t0, 1), ejemplos=ejemplos)
    return res


def arboles(n, k):
    t0 = time.time()
    T = list(nx.nonisomorphic_trees(n))
    P = np.stack([padres_desde_arbol_nx(t) for t in T])
    V = uk_lote(P, k)
    colis = agrupar(V)
    base, _, _ = base_arboles(n, k)
    ejemplos = []
    for grupo in colis[:20]:
        ejemplos.append(dict(padres=[P[i].tolist() for i in grupo],
                             grados=[sorted((d for _, d in T[i].degree()), reverse=True) for i in grupo]))
    return dict(n=n, k=k, arboles=len(T), dim=len(base), grupos_con_colision=len(colis),
                arboles_en_colision=sum(len(g) for g in colis),
                pares_no_isomorfos=sum(len(g) * (len(g) - 1) // 2 for g in colis),
                segundos=round(time.time() - t0, 1), ejemplos=ejemplos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("que", choices=["grafos", "arboles"])
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[3])
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    casos = [(n, None) for n in args.n] if args.que == "grafos" else [(n, k) for n in args.n for k in args.k]
    for n, k in casos:
        if args.que == "grafos":
            r = grafos(n)
            nombre = f"grafos_n{n}.json"
            print(f"grafos n={n}: {r['grafos_conexos']} conexos | grupos con colision {r['grupos_con_colision']} "
                  f"| pares no isomorfos {r['pares_no_isomorfos']} | {r['segundos']}s", flush=True)
        else:
            r = arboles(n, k)
            nombre = f"arboles_n{n}_k{k}.json"
            print(f"arboles n={n} k={k}: {r['arboles']} arboles | grupos con colision {r['grupos_con_colision']} "
                  f"| pares no isomorfos {r['pares_no_isomorfos']} | {r['segundos']}s", flush=True)
        (SALIDA / nombre).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
