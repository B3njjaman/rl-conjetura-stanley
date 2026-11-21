"""Azar con memoria: arboles sueltos con padres uniformes, y los choques de U_3 se buscan en TODO
el historial (no dentro de cada sesion de dos arboles, como hace el Enfoque 2).

Mismo presupuesto que una corrida de H3: 3.000 iteraciones x 1.000 sesiones x 2 arboles = 6e6.
Los pares que existen salen de la verdad de terreno (resultados/verdad_terreno/arboles_n{n}_k3.json).
Un par cuenta como encontrado cuando ya salieron sus dos arboles (en cualquier momento).
Sin redes: solo numpy + Numba.

    python scripts/verdad_terreno/azar_con_memoria.py . 101,102,...,110 resultados/verdad_terreno/azar_con_memoria_k3.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
SEMILLAS = [int(s) for s in sys.argv[2].split(",")]
sys.path.insert(0, str(REPO))
from stanley_rl.upoly import uk_hash_lote, uk_lote  # noqa: E402
from stanley_rl.enfoque2 import canon_arbol  # noqa: E402

PRESUPUESTO = 6_000_000
BLOQUE = 250_000
K = 3
out = {}
for n in (16, 17, 18):
    vt = json.load(open(REPO / "resultados" / "verdad_terreno" / f"arboles_n{n}_k{K}.json"))
    pares = [[np.array(p, dtype=np.int64) for p in ej["padres"]] for ej in vt["ejemplos"]]
    objetivo = {}  # canon -> (hash, vector)
    for par in pares:
        for p in par:
            objetivo[canon_arbol(tuple(p.tolist()))] = (int(uk_hash_lote(p[None, :], K)[0]), uk_lote(p[None, :], K)[0])
    hashes = {h for h, _ in objetivo.values()}
    canon_par = [tuple(canon_arbol(tuple(p.tolist())) for p in par) for par in pares]
    res = []
    for s in SEMILLAS:
        rng = np.random.default_rng(s)
        vistos, primero, todos, t0 = set(), None, None, time.time()
        hechos = 0
        while hechos < PRESUPUESTO:
            m = min(BLOQUE, PRESUPUESTO - hechos)
            # padres[i] uniforme en {0, ..., i-1}: el mismo azar de la linea base de H3
            P = np.zeros((m, n), dtype=np.int64)
            for i in range(1, n):
                P[:, i] = rng.integers(0, i, size=m)
            H = uk_hash_lote(P, K)
            for j in np.nonzero(np.isin(H, np.fromiter(hashes, dtype=np.uint64)))[0]:
                c = canon_arbol(tuple(P[j].tolist()))
                if c in objetivo and np.array_equal(uk_lote(P[j][None, :], K)[0], objetivo[c][1]):
                    if c not in vistos:
                        vistos.add(c)
                        enc = sum(all(x in vistos for x in cp) for cp in canon_par)
                        if enc >= 1 and primero is None:
                            primero = hechos + int(j) + 1
                        if enc == len(pares) and todos is None:
                            todos = hechos + int(j) + 1
            hechos += m
            if todos is not None:
                break
        enc = sum(all(x in vistos for x in cp) for cp in canon_par)
        r = dict(semilla=s, pares_encontrados=enc, de=len(pares), arboles_hasta_primer_par=primero,
                 arboles_hasta_todos=todos, arboles_usados=hechos, segundos=round(time.time() - t0, 1))
        print(f"n={n}", r, flush=True)
        res.append(r)
    out[n] = res
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
