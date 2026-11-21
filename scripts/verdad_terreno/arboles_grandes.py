"""Verdad de terreno del Enfoque 2 para n grandes (hasta 24): todos los pares de arboles no
isomorfos con el mismo U restringido U_k, en cascada. No usa redes neuronales.

    python scripts/verdad_terreno/arboles_grandes.py --n 20 21 22 23 24 --kmax 8

1. Se enumeran todos los arboles (stanley_rl/arboles.py) y se calcula directamente un hash de 64 bits
   del vector U_3 de cada uno (upoly.uk_hash_lote, sin armar el vector).
2. Los arboles con hash repetido se recalculan y se agrupan por el vector exacto (un hash repetido
   por azar no produce un falso positivo).
3. Para k = 4, 5, ... se recalcula U_k solo dentro de los grupos del nivel anterior (si dos arboles
   tienen el mismo U_k, tambien tienen el mismo U_{k-1}). Para cada par se informa el mayor k
   alcanzado.

Deja resultados/verdad_terreno/arboles_grandes_n<n>.json.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from stanley_rl.arboles import arboles  # noqa: E402
from stanley_rl.enfoque2 import canon_arbol  # noqa: E402
from stanley_rl.upoly import uk_hash_lote, uk_lote  # noqa: E402

SALIDA = RAIZ / "resultados" / "verdad_terreno"


def numero_de_arboles(n):
    import networkx as nx
    return nx.number_of_nonisomorphic_trees(n)


def agrupar_exacto(P, idx, k):
    """Subgrupos de idx con el mismo U_k exacto (solo los de 2 o mas)."""
    V = uk_lote(P[idx], k)
    g = defaultdict(list)
    for i, v in zip(idx, V):
        g[v.tobytes()].append(int(i))
    return [np.array(x) for x in g.values() if len(x) > 1]


def resumen_nivel(grupos):
    return dict(grupos=len(grupos), arboles=int(sum(len(g) for g in grupos)),
                pares=int(sum(len(g) * (len(g) - 1) // 2 for g in grupos)),
                mayor_grupo=int(max((len(g) for g in grupos), default=0)))


def correr(n, kmax, bloque):
    t0 = time.time()
    total = numero_de_arboles(n)
    P = np.empty((total, n), dtype=np.int8)
    h = np.empty(total, dtype=np.uint64)
    pos = 0
    for B in arboles(n, bloque=bloque):
        h[pos:pos + len(B)] = uk_hash_lote(B, 3)
        P[pos:pos + len(B)] = B
        pos += len(B)
    assert pos == total, (pos, total)
    t_enum = time.time() - t0
    # hashes repetidos -> candidatos
    orden = np.argsort(h, kind="stable")
    hs = h[orden]
    rep = np.zeros(total, dtype=bool)
    if total > 1:
        igual = hs[1:] == hs[:-1]
        rep[1:] |= igual
        rep[:-1] |= igual
    cand = np.sort(orden[rep])
    niveles = {}
    grupos = agrupar_exacto(P, cand, 3) if len(cand) else []
    niveles[3] = resumen_nivel(grupos)
    k_alcanzado = {}                          # (i, j) -> mayor k con U_k igual
    for g in grupos:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                k_alcanzado[(int(g[a]), int(g[b]))] = 3
    ejemplos = {}
    k = 3
    while grupos and k < min(kmax, n - 1):
        k += 1
        nuevos = []
        for g in grupos:
            nuevos += agrupar_exacto(P, g, k)
        grupos = nuevos
        niveles[k] = resumen_nivel(grupos)
        for g in grupos:
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    k_alcanzado[(int(g[a]), int(g[b]))] = k
        if grupos and len(grupos) <= 200:
            ejemplos[k] = [[P[i].astype(int).tolist() for i in g] for g in grupos]
    hist = defaultdict(int)
    for v in k_alcanzado.values():
        hist[v] += 1
    # chequeo: los pares de cada grupo de verdad no son isomorfos
    for kk, gs in ejemplos.items():
        for g in gs:
            assert len({canon_arbol(tuple(p)) for p in g}) == len(g)
    res = dict(n=n, arboles=total, kmax_revisado=k, niveles={str(a): b for a, b in niveles.items()},
               pares_por_mayor_k={str(a): b for a, b in sorted(hist.items())},
               segundos_enumeracion_u3=round(t_enum, 1), segundos=round(time.time() - t0, 1),
               ejemplos={str(a): b for a, b in ejemplos.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--bloque", type=int, default=250_000)
    a = ap.parse_args()
    if a.kmax < 3 or a.bloque < 1:
        ap.error("--kmax >= 3 (el primer filtro es U_3) y --bloque >= 1")
    SALIDA.mkdir(parents=True, exist_ok=True)
    for n in a.n:
        r = correr(n, a.kmax, a.bloque)
        (SALIDA / f"arboles_grandes_n{n}.json").write_text(json.dumps(r, indent=1))
        print(f"n={n}: {r['arboles']} arboles | " +
              " | ".join(f"U_{k}: {v['pares']} pares" for k, v in r["niveles"].items()) +
              f" | {r['segundos']}s", flush=True)


if __name__ == "__main__":
    main()
