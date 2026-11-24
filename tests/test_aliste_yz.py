"""Control externo del U_k de arboles: los pares Y_{k,l}, Z_{k,l} de Aliste-Prieto, de Mier y Zamora
(arXiv:2002.08186, ec. 14 y 16, Teorema 12): tienen el mismo U_{k+l+2} y distinto U_{k+l+3}.

    A_0 = camino de 3 vertices con raiz en el centro,  B_0 = el mismo con raiz en una hoja
    A_k = A_{k-1} . B_{k-1},   B_k = B_{k-1} . A_{k-1}
    G . H   = union disjunta + arista entre las raices; la raiz es la de G
    G (.) H = union disjunta identificando las raices; la raiz es el vertice identificado
    Y_{k,l} = (A_k (.) A_l) . (B_k (.) B_l),   Z_{k,l} = (A_l (.) B_k) . (B_l (.) A_k)

No usa redes neuronales.
"""
import json
from pathlib import Path

import networkx as nx
import pytest

from stanley_rl.enfoque2 import canon_arbol
from stanley_rl.upoly import padres_desde_arbol_nx, uk_arbol

RAIZ = Path(__file__).resolve().parents[1]


def _p3(raiz_en_centro):
    G = nx.path_graph(3)
    return G, (1 if raiz_en_centro else 0)


def _punto(a, b):
    """G . H: arista entre raices, raiz de G."""
    (G, rg), (H, rh) = a, b
    U = nx.disjoint_union(G, H)
    U.add_edge(rg, len(G) + rh)
    return U, rg


def _identificar(a, b):
    """G (.) H: se identifican las raices."""
    (G, rg), (H, rh) = a, b
    U = nx.disjoint_union(G, H)
    U = nx.contracted_nodes(U, rg, len(G) + rh, self_loops=False)
    U = nx.convert_node_labels_to_integers(U, label_attribute="viejo")
    nueva = next(v for v, d in U.nodes(data=True) if d["viejo"] == rg)
    return nx.Graph(U.edges()), nueva


def _AB(k):
    A, B = _p3(True), _p3(False)
    for _ in range(k):
        A, B = _punto(A, B), _punto(B, A)
    return A, B


def par_yz(k, l):
    Ak, Bk = _AB(k)
    Al, Bl = _AB(l)
    Y = _punto(_identificar(Ak, Al), _identificar(Bk, Bl))[0]
    Z = _punto(_identificar(Al, Bk), _identificar(Bl, Ak))[0]
    return Y, Z


def _padres(T):
    return padres_desde_arbol_nx(nx.convert_node_labels_to_integers(T))


@pytest.mark.parametrize("k,l", [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1), (1, 2)])
def test_teorema_12(k, l):
    Y, Z = par_yz(k, l)
    n = 6 * (2 ** k + 2 ** l) - 2
    assert len(Y) == len(Z) == n and nx.is_tree(Y) and nx.is_tree(Z)
    assert not nx.is_isomorphic(Y, Z)
    pY, pZ = _padres(Y), _padres(Z)
    m = k + l + 2
    assert list(uk_arbol(pY, m)) == list(uk_arbol(pZ, m)), "deberian tener el mismo U_m"
    assert list(uk_arbol(pY, m + 1)) != list(uk_arbol(pZ, m + 1)), "deberian diferir en U_{m+1}"


@pytest.mark.parametrize("n,k,kl", [(10, 2, (0, 0)), (16, 3, (1, 0))])
def test_verdad_terreno_minima_es_el_par_yz(n, k, kl):
    """El unico par que encontro la enumeracion completa es el Y/Z del paper (como dice el paper,
    citando a Smith, Smith y Tian)."""
    d = json.loads((RAIZ / "resultados" / "verdad_terreno" / f"arboles_n{n}_k{k}.json").read_text())
    assert d["pares_no_isomorfos"] == 1
    mio = {canon_arbol(tuple(p)) for p in d["ejemplos"][0]["padres"]}
    Y, Z = par_yz(*kl)
    suyo = {canon_arbol(tuple(int(x) for x in _padres(T))) for T in (Y, Z)}
    assert mio == suyo


def test_primer_par_u4_es_y11_z11():
    """La enumeracion completa de n = 22 (Kaggle) da un solo par con igual U_4: el Y_{1,1}/Z_{1,1}."""
    d = json.loads((RAIZ / "resultados" / "verdad_terreno" / "arboles_grandes_n22.json").read_text())
    assert d["niveles"]["4"]["pares"] == 1 and d["niveles"]["5"]["pares"] == 0
    mio = {canon_arbol(tuple(p)) for p in d["ejemplos"]["4"][0]}
    suyo = {canon_arbol(tuple(int(x) for x in _padres(T))) for T in par_yz(1, 1)}
    assert mio == suyo
