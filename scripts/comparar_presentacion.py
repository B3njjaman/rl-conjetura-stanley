"""Compara los pares de la presentacion de la tesis con la verdad de terreno.

Las aristas se leyeron a ojo de las figuras (ResultadoN6.png, VisualizacionN7.png, ArbolN12k3.png,
ArbolN24k3.png, ArbolN24k4.png): las de N = 24 son aproximadas.

    python scripts/comparar_presentacion.py .
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
from stanley_rl.upoly import u_signed_nx, uk_arbol, uk_lote, padres_desde_arbol_nx  # noqa: E402
from stanley_rl.arboles import arboles  # noqa: E402

VT = REPO / "resultados" / "verdad_terreno"


def grafo(n, aristas):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(aristas)
    return G


# Pares de grafos con ciclos de la presentacion (ResultadoN6.png, VisualizacionN7.png)
pres = {
    6: (grafo(6, [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5)]),
        grafo(6, [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])),
    7: (grafo(7, [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 3), (3, 6)]),
        grafo(7, [(0, 1), (1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (3, 5), (4, 6)])),
}
for n, (G1, G2) in pres.items():
    u1, u2 = u_signed_nx(G1), u_signed_nx(G2)
    iso = nx.is_isomorphic(G1, G2)
    vt = json.load(open(VT / f"grafos_n{n}.json"))
    en_lista = False
    for ej in vt["ejemplos"]:
        grupo = [grafo(n, [tuple(e) for e in g]) for g in ej["grafos"]]
        if any(nx.is_isomorphic(G1, A) for A in grupo) and any(nx.is_isomorphic(G2, B) for B in grupo):
            en_lista = True
    print(f"N={n}: mismo U(x,0)={np.array_equal(u1, u2)}, isomorfos={iso}, "
          f"conexos={nx.is_connected(G1) and nx.is_connected(G2)}, "
          f"esta entre los {len(vt['ejemplos'])} de la verdad de terreno={en_lista}")

# Par de arboles N=12, k=3 de la presentacion (ArbolN12k3.png)
T1 = grafo(12, [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (1, 9), (9, 8), (9, 10), (10, 11)])
T2 = grafo(12, [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (5, 7), (7, 8), (8, 9), (7, 10), (10, 11)])
assert nx.is_tree(T1) and nx.is_tree(T2)
v1 = np.asarray(uk_arbol(tuple(padres_desde_arbol_nx(T1)), 3), dtype=float)
v2 = np.asarray(uk_arbol(tuple(padres_desde_arbol_nx(T2)), 3), dtype=float)
print(f"N=12 k=3 presentacion: isomorfos={nx.is_isomorphic(T1, T2)}, distancia L2={np.linalg.norm(v1 - v2):.4f}")

# Minimo exacto de la distancia L2 entre los U_3 de arboles no isomorfos con n=12
P = np.vstack(list(arboles(12)))
V = uk_lote(P, 3).astype(float)
D = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)
m = D.min()
npares = int((np.isclose(D, m)).sum() // 2)
print(f"N=12 k=3 exhaustivo: {len(P)} arboles, distancia minima={m:.4f} (sqrt6={np.sqrt(6):.4f}), "
      f"pares en el minimo={npares}")


def arbol_letras(aristas):
    nombres = sorted({v for e in aristas for v in e}, key=lambda s: (len(s), s) if isinstance(s, str) else (0, s))
    idx = {v: i for i, v in enumerate(nombres)}
    T = grafo(len(nombres), [(idx[a], idx[b]) for a, b in aristas])
    assert nx.is_tree(T), "no es arbol"
    return T


def dist(T1, T2, k):
    a = np.asarray(uk_arbol(tuple(padres_desde_arbol_nx(T1)), k), dtype=float)
    b = np.asarray(uk_arbol(tuple(padres_desde_arbol_nx(T2)), k), dtype=float)
    return np.linalg.norm(a - b)


# ArbolN24k3.png
A3 = arbol_letras([(1, 2), (2, 3), (3, 4), (5, 6), (6, 4), (4, 7), (7, 8), (8, 9), (9, 10), (9, 11), (7, 12), (12, 13),
                   (12, 14), (12, 15), (15, 16), (15, 17), (15, 18), (12, 19), (19, 20), (19, 21), (21, 22), (22, 23),
                   (22, 24)])
B3 = arbol_letras([tuple(e) for e in "ab bc cd de df fg fh hi ij hk kl lm mn no mp lq qr ls lt ku uv uw ux".split()])
# ArbolN24k4.png
A4 = arbol_letras([(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (8, 9), (7, 10), (10, 11), (10, 12), (7, 13),
                   (13, 14), (13, 15), (15, 16), (16, 17), (17, 18), (15, 19), (19, 20), (19, 21), (21, 22), (21, 23),
                   (23, 24)])
B4 = arbol_letras([tuple(e) for e in "ac bc ce de ef fg gh fi ik kj il im fn no np pq qr ps pt tu tv vw wx".split()])
for nombre, (T1, T2, k) in {"N=24 k=3": (A3, B3, 3), "N=24 k=4": (A4, B4, 4)}.items():
    print(f"{nombre} presentacion: n={T1.number_of_nodes()},{T2.number_of_nodes()} "
          f"isomorfos={nx.is_isomorphic(T1, T2)}, distancia L2={dist(T1, T2, k):.4f}")
