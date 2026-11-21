import networkx as nx
import numpy as np
import pytest

from stanley_rl.arboles import arboles
from stanley_rl.enfoque2 import canon_arbol
from stanley_rl.upoly import padres_desde_arbol_nx


@pytest.mark.parametrize("n", range(3, 17))
def test_cuenta_igual_a_networkx(n):
    total = sum(len(b) for b in arboles(n, bloque=777))
    assert total == nx.number_of_nonisomorphic_trees(n)


@pytest.mark.parametrize("n", [5, 8, 11, 12])
def test_mismos_arboles_que_networkx(n):
    P = np.concatenate(list(arboles(n)))
    assert (P[:, 1:] < np.arange(1, n)).all()            # padres[i] < i
    mios = {canon_arbol(tuple(p.tolist())) for p in P}
    suyos = {canon_arbol(tuple(padres_desde_arbol_nx(T).tolist())) for T in nx.nonisomorphic_trees(n)}
    assert len(mios) == len(P) == len(suyos) and mios == suyos
