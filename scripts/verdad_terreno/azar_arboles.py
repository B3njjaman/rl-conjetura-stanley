"""Probabilidad exacta de que la generacion con padres uniformes (padres[i] ~ U{0..i-1}, es decir un
arbol recursivo al azar) produzca en una sesion un par (T1, T2) no isomorfo con el mismo U_k.

    P(T ~= A) = n / |Aut(A)| * sum_r 1 / prod_v sz_r(v)

(numero de etiquetados crecientes de A con raiz r = n! / prod sz_r, dividido por (n-1)! arboles
recursivos). Es la linea base "sin aprendizaje" de H3. La red sin entrenar (lr = 0) no es exactamente
uniforme: el control empirico es el lote h3_control.

    python scripts/verdad_terreno/azar_arboles.py --n 16 17 18 --k 3
"""
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import networkx as nx

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from stanley_rl.enfoque2 import canon_arbol  # noqa: E402


def _arbol(padres):
    return nx.Graph([(i, int(padres[i])) for i in range(1, len(padres))])


def _aut_con_raiz(T, v, p):
    hijos = [_aut_con_raiz(T, w, v) for w in T[v] if w != p]
    aut = 1
    cuenta = {}
    for codigo, a in hijos:
        aut *= a
        cuenta[codigo] = cuenta.get(codigo, 0) + 1
    for c in cuenta.values():
        aut *= math.factorial(c)
    return "(" + "".join(sorted(c for c, _ in hijos)) + ")", aut


def automorfismos(T):
    centro = nx.center(T)
    if len(centro) == 1:
        return _aut_con_raiz(T, centro[0], None)[1]
    a, b = centro
    ca, aa = _aut_con_raiz(T, a, b)
    cb, ab = _aut_con_raiz(T, b, a)
    return aa * ab * (2 if ca == cb else 1)


def _tamanos(T, r):
    padre = {r: None}
    orden = [r]
    for v in orden:
        for w in T[v]:
            if w not in padre:
                padre[w] = v
                orden.append(w)
    sz = dict.fromkeys(T, 1)
    for v in reversed(orden[1:]):
        sz[padre[v]] += sz[v]
    return sz


def prob_forma(T):
    n = len(T)
    s = Fraction(0)
    for r in T:
        prod = 1
        for x in _tamanos(T, r).values():
            prod *= x
        s += Fraction(1, prod)
    return Fraction(n, automorfismos(T)) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sesiones", type=float, default=3e6, help="sesiones por corrida (1000 x 3000 iter)")
    a = ap.parse_args()
    out = {}
    for n in a.n:
        d = json.loads((RAIZ / "resultados" / "verdad_terreno" / f"arboles_n{n}_k{a.k}.json").read_text())
        p = Fraction(0)
        detalle = []
        for g in d["ejemplos"]:
            ps = [prob_forma(_arbol(x)) for x in g["padres"]]
            S = sum(ps)
            p += S * S - sum(x * x for x in ps)        # pares ordenados (T1, T2) no isomorfos del grupo
            detalle.append([float(x) for x in ps])
        pf = float(p)
        esperado = pf * a.sesiones
        out[n] = dict(p_por_sesion=pf, pares=d["pares_no_isomorfos"], grupos=len(d["ejemplos"]),
                      esperado_por_corrida=esperado, p_al_menos_uno=1 - math.exp(-esperado),
                      prob_formas=detalle)
        print(f"n={n} k={a.k}: {d['pares_no_isomorfos']} pares | p por sesion = {pf:.3e} | "
              f"esperado en {a.sesiones:.0e} sesiones = {esperado:.3f} | P(>=1) = {1 - math.exp(-esperado):.3f}")
    (RAIZ / "resultados" / "verdad_terreno" / f"azar_arboles_k{a.k}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
