# Verdad de terreno para N=19: el minimo de lambda1+mu sobre grafos conexos se alcanza en un
# arbol (sacar una arista manteniendo conexidad baja lambda1 y no sube mu), asi que basta
# enumerar los 317 955 arboles no isomorfos de 19 vertices y evaluar el score de cada uno.
# Deja en resultados/01_wagner_conj21/verdad_terreno_arboles19/: resumen.json, top_arboles.json,
# histograma.json (versionados) y scores.npy (local, ignorado por git).
import json, math, sys, time
from pathlib import Path
import numpy as np, networkx as nx, torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from nb01_core import N, numero_matching

RAIZ = Path(__file__).resolve().parents[2]
SAL = RAIZ / "resultados" / "01_wagner_conj21" / "verdad_terreno_arboles19"
SAL.mkdir(parents=True, exist_ok=True)
base = math.sqrt(N - 1) + 1
t0 = time.time()
scores = []
top = []
for k, T in enumerate(nx.nonisomorphic_trees(N)):
    A = nx.to_numpy_array(T, dtype=np.int64)
    lam1 = float(np.max(np.abs(np.linalg.eigvalsh(A.astype(np.float64)))))
    mu = int(numero_matching(A))
    s = base - lam1 - mu
    scores.append(s)
    if s > -0.5:
        top.append(dict(score=s, lambda1=lam1, mu=mu, grados=sorted((d for _, d in T.degree()), reverse=True),
                        aristas=sorted(T.edges())))
    if k % 50000 == 0:
        print(f"{k} arboles, {time.time() - t0:.0f}s", flush=True)
scores = np.array(scores)
top.sort(key=lambda d: -d["score"])
resumen = dict(N=N, cota_conjetura=base, total_arboles=int(len(scores)), score_maximo=float(scores.max()),
               arboles_score_positivo=int((scores > 0).sum()), arboles_score_mayor_m0_1=int((scores > -0.1).sum()),
               arboles_score_mayor_m0_5=int((scores > -0.5).sum()), arboles_score_mayor_m1=int((scores > -1).sum()),
               score_mediana=float(np.median(scores)), segundos=round(time.time() - t0))
json.dump(resumen, open(SAL / "resumen.json", "w"), indent=1)
json.dump(top, open(SAL / "top_arboles.json", "w"), indent=1)
bordes = np.arange(math.floor(scores.min()), 0.3, 0.1)
cuentas, _ = np.histogram(scores, bins=bordes)
json.dump(dict(bordes=bordes.round(3).tolist(), cuentas=cuentas.tolist()), open(SAL / "histograma.json", "w"))
np.save(SAL / "scores.npy", scores)
print(json.dumps(resumen, indent=1))
