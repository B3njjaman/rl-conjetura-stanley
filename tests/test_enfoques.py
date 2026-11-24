"""Tests de los Enfoques 1 y 2 sin redes neuronales: recompensa, codificacion, estados y seleccion.

La generacion de sesiones se prueba con una funcion de probabilidades fija (no una red).
"""
import itertools
import random

import networkx as nx
import numpy as np

from stanley_rl import enfoque1 as e1
from stanley_rl import enfoque2 as e2
from stanley_rl.ce import sel_wagner, sel_percentil
from stanley_rl.upoly import padres_desde_arbol_nx


def palabra_de(G, N):
    return [1 if G.has_edge(i, j) else 0 for i in range(N) for j in range(i + 1, N)]


# ---- Enfoque 1 ---------------------------------------------------------------------------
CORBATA = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)])
DIAMANTE_COLA = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])


def test_recompensa_par_de_stanley_vale_1():
    r = e1.Recompensa(5)
    acc = np.array([palabra_de(CORBATA, 5) + palabra_de(DIAMANTE_COLA, 5)], dtype=np.int8)
    assert r(acc)[0] == 1.0
    assert len(r.encontrados) == 1


def test_recompensa_isomorfos_y_desconexos():
    r = e1.Recompensa(5)
    perm = nx.relabel_nodes(CORBATA, {0: 4, 1: 0, 2: 3, 3: 1, 4: 2})
    iso = palabra_de(CORBATA, 5) + palabra_de(perm, 5)
    desc = palabra_de(CORBATA, 5) + palabra_de(nx.Graph([(0, 1), (2, 3), (3, 4), (2, 4)]).copy(), 5)
    rec = r(np.array([iso, desc], dtype=np.int8))
    assert list(rec) == [e1.NEG_INF, e1.NEG_INF]
    assert r.encontrados == {}


def test_recompensa_igual_a_la_del_notebook():
    # recompensa original: iso -> -1e6, desconexo -> -1e6, si no 1/(1+mse) con el U con signo
    from test_upoly import vector_original
    rng = random.Random(3)
    N = 5
    r = e1.Recompensa(N)
    for _ in range(150):
        G1 = nx.gnp_random_graph(N, rng.uniform(0.3, 0.9), seed=rng.randrange(10 ** 9))
        G2 = nx.gnp_random_graph(N, rng.uniform(0.3, 0.9), seed=rng.randrange(10 ** 9))
        if nx.is_isomorphic(G1, G2) or not nx.is_connected(G1) or not nx.is_connected(G2):
            esperado = -1e6
        else:
            v1, v2 = vector_original(G1), vector_original(G2)
            esperado = 1 / (1 + np.mean((v1 - v2) ** 2.0))
        acc = np.array([palabra_de(G1, N) + palabra_de(G2, N)], dtype=np.int8)
        assert abs(r(acc)[0] - esperado) < 1e-12


def test_canon_es_invariante_y_separa():
    rng = random.Random(1)
    for _ in range(30):
        G = nx.gnp_random_graph(6, 0.5, seed=rng.randrange(10 ** 9))
        H = nx.relabel_nodes(G, dict(zip(range(6), rng.sample(range(6), 6))))
        m = lambda X: np.array([sum(1 << v for v in X[u]) for u in range(6)], dtype=np.int64)
        assert e1.canon(m(G), 6) == e1.canon(m(H), 6)
    G7 = [g for g in nx.graph_atlas_g() if g.number_of_nodes() == 6]
    codigos = {int(e1.canon(np.array([sum(1 << v for v in g[u]) for u in range(6)], dtype=np.int64), 6)) for g in G7}
    assert len(codigos) == len(G7)            # 156 grafos de 6 vertices, 156 formas canonicas


def test_generar_y_estados_coinciden():
    N = 4
    M, L, obs = e1.dimensiones(N)
    vistos = []

    def prob_fija(estado):
        vistos.append(estado.copy())
        return np.full(len(estado), 0.5)

    acc = e1.generar(prob_fija, 7, L, np.random.default_rng(0))
    X, y = e1.estados_de(acc, L)
    X = X.reshape(7, L, obs)
    for t in range(L):
        assert np.array_equal(X[:, t, :], vistos[t])
    assert np.array_equal(y, acc.reshape(-1))


# ---- Enfoque 2 ---------------------------------------------------------------------------
def test_canon_arbol():
    rng = random.Random(5)
    for _ in range(40):
        T = nx.random_labeled_tree(11, seed=rng.randrange(10 ** 9))
        H = nx.relabel_nodes(T, dict(zip(range(11), rng.sample(range(11), 11))))
        H = nx.convert_node_labels_to_integers(H)
        a = e2.canon_arbol(tuple(padres_desde_arbol_nx(T).tolist()))
        b = e2.canon_arbol(tuple(padres_desde_arbol_nx(H).tolist()))
        assert a == b
    todos = list(nx.nonisomorphic_trees(10))
    assert len({e2.canon_arbol(tuple(padres_desde_arbol_nx(t).tolist())) for t in todos}) == len(todos)


def test_recompensa_arboles():
    n, k = 10, 2
    # par conocido con igual U_2 (verdad de terreno n=10, k=2): se busca a la fuerza
    T = list(nx.nonisomorphic_trees(n))
    P = [padres_desde_arbol_nx(t) for t in T]
    from stanley_rl.upoly import uk_arbol
    V = [uk_arbol(p, k).tobytes() for p in P]
    i, j = next((i, j) for i, j in itertools.combinations(range(len(T)), 2) if V[i] == V[j])
    acc = np.array([list(P[i][1:]) + list(P[j][1:]), list(P[i][1:]) + list(P[i][1:])])
    r = e2.Recompensa(n, k)
    rec, dist = r(acc)
    assert rec[0] == 1.0 and rec[1] == e2.PEN_ISO and len(r.encontrados) == 1


def test_generar_arboles_y_estados():
    n = 6
    L, obs = e2.dimensiones(n)
    vistos = []

    def logit_fijo(estado):
        vistos.append(estado.copy())
        return np.zeros((len(estado), n))

    acc = e2.generar(logit_fijo, 9, n, np.random.default_rng(1))
    for t in range(L):                                    # el padre siempre es valido
        assert acc[:, t].max() < e2.vertice_del_paso(t, n)
    X, y, masc = e2.estados_de(acc, n)
    X = X.reshape(9, L, obs)
    for t in range(L):
        assert np.array_equal(X[:, t, :], vistos[t])
    assert masc.reshape(9, L, n)[0, 0].sum() == 1 and masc.reshape(9, L, n)[0, n - 2].sum() == n - 1


# ---- seleccion ---------------------------------------------------------------------------
def test_seleccion_wagner_con_empates():
    REC = np.array([5.0] * 20 + [1.0] * 80)
    n_ses = 100
    assert sel_percentil(REC, 93).sum() == 20             # sin tope entran todos los empatados
    assert sel_wagner(REC, 93, n_ses, "super").sum() == 7  # con tope: n_ses * 7% = 7
    assert sel_wagner(REC, 93, n_ses, "elite").sum() == 7
