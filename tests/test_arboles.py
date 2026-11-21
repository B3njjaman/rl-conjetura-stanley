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


@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_probabilidad_de_forma_exacta(n):
    """La formula de azar_arboles.py contra los (n-1)! arboles recursivos (padres uniformes)."""
    import math
    import sys
    from fractions import Fraction
    from itertools import product
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "verdad_terreno"))
    from azar_arboles import _arbol, prob_forma
    cuenta, rep = {}, {}
    for pad in product(*[range(i) for i in range(1, n)]):
        P = (0,) + pad
        c = canon_arbol(P)
        cuenta[c] = cuenta.get(c, 0) + 1
        rep.setdefault(c, P)
    total = math.factorial(n - 1)
    assert len(cuenta) == nx.number_of_nonisomorphic_trees(n)
    for c, m in cuenta.items():
        assert prob_forma(_arbol(rep[c])) == Fraction(m, total)
