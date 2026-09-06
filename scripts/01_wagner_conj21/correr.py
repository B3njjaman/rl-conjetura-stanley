# Corre UNA instancia del algoritmo del notebook 01 y deja todo en resultados/01_wagner_conj21/<nombre>/
#   python correr.py --modo notebook   --seed 0 --iters 1000
#   python correr.py --modo wagner_fit --seed 0 --iters 3000 --tiempo-max 3600
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, networkx as nx

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

ap = argparse.ArgumentParser()
ap.add_argument("--modo", default="notebook", choices=["notebook", "wagner_fit"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--tiempo-max", type=float, default=None, help="segundos; corta la corrida aunque falten iteraciones")
ap.add_argument("--hilos", type=int, default=2, help="hilos de torch (para correr varias instancias a la vez)")
ap.add_argument("--nombre", default=None, help="subcarpeta en resultados/01_wagner_conj21 (default: modo_seedN)")
args = ap.parse_args()

torch.set_num_threads(args.hilos)
from variantes import *          # importa tambien todo nb01_core (N, MYN, calcScore, ...)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
nombre = args.nombre or f"{args.modo}_seed{args.seed}"
carpeta = RAIZ / "resultados" / "01_wagner_conj21" / nombre
carpeta.mkdir(parents=True, exist_ok=True)
flog = open(carpeta / "log.txt", "w", buffering=1)


def log(msg):
    flog.write(msg + "\n")
    if not msg.startswith("iter") or int(msg.split()[1]) % 50 == 0:
        print(f"[{nombre}] {msg}", flush=True)


log(f"modo={args.modo} seed={args.seed} iters={args.iters} tiempo_max={args.tiempo_max} hilos={args.hilos} device={device}")
res = entrenar_variante(args.iters, modo=args.modo, carpeta=str(carpeta), tiempo_max=args.tiempo_max, log=log)

# historial por iteracion (CSV: se versiona, a diferencia de los .pkl/.pt)
with open(carpeta / "historial.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res["historial"][0].keys()))
    w.writeheader()
    w.writerows(res["historial"])
json.dump(res["linea_tiempo"], open(carpeta / "linea_tiempo.json", "w"))


# mejor grafo decodificado y verificado con networkx (independiente de la recompensa en numba)
def grafo(palabra):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            if palabra[c] == 1:
                G.add_edge(i, j)
            c += 1
    return G


G = grafo(res["mejor"]["palabra"])
conexo = nx.is_connected(G)
lam1 = float(max(abs(np.linalg.eigvalsh(nx.to_numpy_array(G))))) if conexo else None
mu = len(nx.max_weight_matching(G, maxcardinality=True))
score_nx = (math.sqrt(N - 1) + 1 - lam1 - mu) if conexo else None
mejor = dict(iter=res["mejor"]["iter"], score=res["mejor"]["score"], score_networkx=score_nx, lambda1=lam1, mu=mu,
             conexo=conexo, arbol=nx.is_tree(G), n_aristas=G.number_of_edges(),
             grados=sorted((d for _, d in G.degree()), reverse=True), aristas=sorted(G.edges()),
             palabra=res["mejor"]["palabra"])
json.dump(mejor, open(carpeta / "mejor_grafo.json", "w"), indent=1)

resumen = dict(nombre=nombre, modo=args.modo, seed=args.seed, iters_pedidas=args.iters,
               iters_corridas=len(res["historial"]), tiempo_max=args.tiempo_max, segundos=round(res["segundos"]),
               mejor_score=res["mejor"]["score"], contraejemplo_encontrado=bool(res["mejor"]["score"] > 0),
               mejor_iter=res["mejor"]["iter"], lambda1=lam1, mu=mu, arbol=mejor["arbol"],
               grados_mayores=mejor["grados"][:5])
json.dump(resumen, open(carpeta / "resumen.json", "w"), indent=1)
log("FIN " + json.dumps(resumen))
flog.close()
