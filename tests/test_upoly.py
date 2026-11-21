"""Tests del U-polinomio. No usan redes neuronales (se pueden correr en el PC).

La referencia es la funcion del codigo original de la tesis (GrafosCompletos_noisomorfos.ipynb,
celda 1), copiada tal cual: recorre los 2^|E| subconjuntos de aristas con networkx.
"""
import itertools
import random

import networkx as nx
import numpy as np
import pytest

from stanley_rl.upoly import (particiones, u_signed_nx, uk_arbol, padres_desde_arbol_nx,
                              base_arboles)


# ---- referencia: codigo original (celda 1 del notebook de la tesis) ----------------------
def get_lambda_partition(subgraph):
    sizes = [len(c) for c in nx.connected_components(subgraph)]
    return tuple(sorted(sizes, reverse=True))


def chromatic_symmetric_y_form(G):
    n = G.number_of_nodes()
    edges = list(G.edges())
    terms = {}
    for A_edges in itertools.chain.from_iterable(
            itertools.combinations(edges, r) for r in range(len(edges) + 1)):
        subG = nx.Graph()
        subG.add_nodes_from(G.nodes())
        subG.add_edges_from(A_edges)
        lam = get_lambda_partition(subG)
        m = len(A_edges)
        k = len(lam)
        rank = n - k
        nullity = m - rank
        coeff = (-1) ** nullity
        terms[lam] = terms.get(lam, 0) + coeff
    return terms


def vector_original(G):
    base = particiones(G.number_of_nodes())
    t = chromatic_symmetric_y_form(G)
    return np.array([t.get(p, 0) for p in base])


# ---- ejemplos a mano ---------------------------------------------------------------------
def test_base_orden_original():
    assert particiones(4) == [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
    assert [len(particiones(n)) for n in (5, 10, 20)] == [7, 42, 627]


def test_estrella_y_camino_de_la_tesis():
    # tesis, ejemplo de K_{1,3} y P_4 (aqui en el orden de base de particiones(4))
    assert list(u_signed_nx(nx.star_graph(3))) == [1, 3, 0, 3, 1]
    assert list(u_signed_nx(nx.path_graph(4))) == [1, 2, 1, 3, 1]


def test_triangulo_distingue_el_signo():
    # (3): A con 2 aristas (+3) y A = E con nulidad 1 (-1)  -> 2 con signo (4 sin signo)
    assert list(u_signed_nx(nx.complete_graph(3))) == [2, 3, 1]


def test_par_de_stanley_5_vertices():
    # corbata (dos triangulos por un vertice) vs diamante con una arista colgando
    corbata = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)])
    diamante_cola = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
    assert not nx.is_isomorphic(corbata, diamante_cola)
    assert list(u_signed_nx(corbata)) == list(u_signed_nx(diamante_cola))


# ---- contra el codigo original -----------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_igual_al_original_grafos_aleatorios(n):
    rng = random.Random(n)
    for _ in range(25):
        p = rng.random()
        G = nx.gnp_random_graph(n, p, seed=rng.randrange(10 ** 9))
        assert list(u_signed_nx(G)) == list(vector_original(G))


def test_igual_al_original_todos_los_grafos_hasta_5():
    for G in nx.graph_atlas_g()[1:]:
        if 2 <= G.number_of_nodes() <= 5:
            assert list(u_signed_nx(G)) == list(vector_original(G))


# ---- arboles: U_k restringido ------------------------------------------------------------
def uk_bruto(T, k):
    """Referencia: quitar cada subconjunto de <= k aristas y mirar las componentes."""
    n = T.number_of_nodes()
    base, _, _ = base_arboles(n, k)
    edges = list(T.edges())
    v = dict.fromkeys(base, 0)
    for j in range(min(k, n - 1) + 1):
        for quitar in itertools.combinations(edges, j):
            H = T.copy()
            H.remove_edges_from(quitar)
            v[get_lambda_partition(H)] += 1
    return np.array([v[p] for p in base])


@pytest.mark.parametrize("n", [2, 3, 5, 8, 10])
def test_uk_arbol_contra_bruto(n):
    rng = random.Random(100 + n)
    for _ in range(10):
        T = nx.random_labeled_tree(n, seed=rng.randrange(10 ** 9)) if n > 1 else nx.empty_graph(1)
        padres = padres_desde_arbol_nx(T)
        for k in range(0, n):
            assert list(uk_arbol(padres, k)) == list(uk_bruto(T, k)), (n, k)


@pytest.mark.parametrize("n", [4, 6, 9])
def test_uk_completo_es_el_u_de_grafos(n):
    # para un arbol, U_{n-1} (todas las aristas) = U con signo (no hay ciclos, no hay signo)
    rng = random.Random(7 * n)
    for _ in range(10):
        T = nx.random_labeled_tree(n, seed=rng.randrange(10 ** 9))
        padres = padres_desde_arbol_nx(T)
        Tr = nx.Graph([(i, int(padres[i])) for i in range(1, n)])
        assert list(uk_arbol(padres, n - 1)) == list(u_signed_nx(Tr))


# ---- U_k incremental (CAMBIO 18) contra la version por combinaciones ----------------------
@pytest.mark.parametrize("n,ks", [(12, range(0, 12)), (18, (2, 3, 4, 5)), (24, (3, 4)), (30, (3, 5))])
def test_uk_incremental_igual_a_combinaciones(n, ks):
    from stanley_rl.upoly import uk_arbol_combinaciones
    rng = random.Random(1000 + n)
    for _ in range(8):
        padres = padres_desde_arbol_nx(nx.random_labeled_tree(n, seed=rng.randrange(10 ** 9)))
        for k in ks:
            assert list(uk_arbol(padres, k)) == list(uk_arbol_combinaciones(padres, k)), (n, k)


def test_uk_lote_y_hash_agrupan_igual():
    from stanley_rl.arboles import arboles
    from stanley_rl.upoly import uk_lote, uk_hash_lote
    for n, k in ((12, 2), (14, 2), (16, 3)):
        P = np.concatenate(list(arboles(n)))
        V = uk_lote(P, k)
        H = uk_hash_lote(P, k)
        por_vector = {}
        for i, v in enumerate(V):
            por_vector.setdefault(v.tobytes(), []).append(i)
        por_hash = {}
        for i, h in enumerate(H):
            por_hash.setdefault(int(h), []).append(i)
        assert sorted(map(tuple, por_vector.values())) == sorted(map(tuple, por_hash.values())), (n, k)
