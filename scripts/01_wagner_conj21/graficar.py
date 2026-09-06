# Figuras "tipo Wagner" para el reporte de la tesis, a partir de lo que dejan correr.py y
# enumerar_arboles.py en resultados/01_wagner_conj21/. Se puede correr las veces que se quiera:
#   python graficar.py
# Por corrida:  recompensa.png, mejor_grafo.png, linea_tiempo.png
# Globales:     comparacion_corridas.png, resumen_corridas.md,
#               verdad_terreno_arboles19/histograma_scores.png y contraejemplo_wagner.png
import csv, json, math
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "resultados" / "01_wagner_conj21"
N = 19
COTA = math.sqrt(N - 1) + 1            # sqrt(18)+1: lo que conjeturaba Aouchiche-Hansen
OPTIMO = COTA - math.sqrt(10) - 2      # +0.0804: el contraejemplo de Wagner (maximo global, ver verdad de terreno)

# colores: un tono por modo (identidad), semilla por estilo de linea
COLOR = {"notebook": "#2a78d6", "wagner_fit": "#eb6834", "otro": "#1baf7a"}
ESTILO = {0: "-", 1: "--", 2: ":", 3: "-."}
GRIS = "#52514e"
ETIQ = {"notebook": "notebook (Adam 1e-2, 1 paso/iter)", "wagner_fit": "fit de Wagner (SGD 1e-4, minilotes)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddcd8",
    "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8,
})


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


def dibujar(G, ax, color, titulo=None):
    pos = nx.kamada_kawai_layout(G)          # mismo layout que usa Wagner (nx.draw_kamada_kawai)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRIS, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=45, node_color=color, edgecolors="white", linewidths=0.8)
    ax.set_axis_off()
    if titulo:
        ax.set_title(titulo)


def lineas_referencia(ax, xmax):
    ax.axhline(0, color=GRIS, lw=0.9, ls="--")
    ax.axhline(OPTIMO, color=GRIS, lw=0.9, ls=":")
    ax.text(xmax, 0, " umbral contraejemplo (0)", va="bottom", ha="right", fontsize=7, color=GRIS)
    ax.text(xmax, OPTIMO, f" grafo de Wagner (+{OPTIMO:.4f})", va="bottom", ha="right", fontsize=7, color=GRIS)


def leer_historial(carpeta):
    with open(carpeta / "historial.csv") as f:
        filas = list(csv.DictReader(f))
    h = {k: np.array([float(r[k]) for r in filas]) for k in filas[0]}
    return h


def figuras_corrida(carpeta):
    resumen = json.load(open(carpeta / "resumen.json"))
    modo, seed = resumen["modo"], resumen["seed"]
    color = COLOR.get(modo, COLOR["otro"])
    h = leer_historial(carpeta)
    it = h["iter"]

    # 1) evolucion de la recompensa (la figura 5.1 de la tesis / las curvas de Wagner)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(it, h["mejor"], color=color, lw=1.6, label="mejor sesion de la iteracion")
    ax.plot(it, h["top100"], color=color, lw=1.2, alpha=0.55, label="promedio de las 100 mejores")
    lineas_referencia(ax, it[-1])
    ax.set_xlabel("iteracion")
    ax.set_ylabel(r"score  $\sqrt{n-1}+1-\lambda_1-\mu$")
    ax.set_title(f"Cross-Entropy en PyTorch, conjetura 2.1 (N={N}): {ETIQ.get(modo, modo)}, semilla {seed}")
    ax.legend(loc="lower right")
    ymin = max(h["mejor"].min(), -14)
    ax.set_ylim(ymin - 0.5, 0.6)
    fig.tight_layout()
    fig.savefig(carpeta / "recompensa.png")
    plt.close(fig)

    # 2) mejor grafo encontrado (la figura 5.2 de la tesis)
    mg = json.load(open(carpeta / "mejor_grafo.json"))
    G = grafo(mg["palabra"])
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    tipo = "arbol" if mg["arbol"] else ("conexo" if mg["conexo"] else "NO conexo")
    dibujar(G, ax, color, f"mejor grafo ({tipo}, iter {mg['iter']})\n"
                          f"$\\lambda_1$={mg['lambda1']:.4f}, $\\mu$={mg['mu']}, score={mg['score']:+.4f}")
    fig.tight_layout()
    fig.savefig(carpeta / "mejor_grafo.png")
    plt.close(fig)

    # 3) linea de tiempo del mejor grafo (como la figura 3 de Wagner)
    lt = json.load(open(carpeta / "linea_tiempo.json"))
    if len(lt) >= 2:
        idx = sorted(set(np.unique(np.geomspace(1, len(lt), num=min(8, len(lt))).astype(int) - 1)) | {0, len(lt) - 1})
        sel = [lt[i] for i in idx][:8]
        fig, axs = plt.subplots(2, 4, figsize=(10, 5.2))
        for ax, d in zip(axs.ravel(), sel):
            dibujar(grafo(d["palabra"]), ax, color, f"iter {d['iter']}: score {d['score']:+.3f}")
        for ax in axs.ravel()[len(sel):]:
            ax.set_axis_off()
        fig.suptitle(f"Mejor grafo a lo largo del entrenamiento: {ETIQ.get(modo, modo)}, semilla {seed}")
        fig.tight_layout()
        fig.savefig(carpeta / "linea_tiempo.png")
        plt.close(fig)
    return resumen, h


def figura_comparacion(corridas):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    xmax = 0
    for resumen, h in corridas:
        modo, seed = resumen["modo"], resumen["seed"]
        color, ls = COLOR.get(modo, COLOR["otro"]), ESTILO.get(seed, "-")
        acum = np.maximum.accumulate(h["mejor"])
        a1.plot(h["iter"], acum, color=color, ls=ls, lw=1.6, label=f"{ETIQ.get(modo, modo)}, semilla {seed}")
        a1.text(h["iter"][-1], acum[-1], f" {acum[-1]:+.3f}", va="center", fontsize=7, color=color)
        a2.plot(h["iter"], h["super_distintos"], color=color, ls=ls, lw=1.4)
        xmax = max(xmax, h["iter"][-1])
    lineas_referencia(a1, xmax)
    a1.set_ylabel("mejor score acumulado")
    a1.set_title(f"Conjetura 2.1 (N={N}): entrenamiento del notebook vs. el fit de Wagner")
    a1.legend(loc="lower right")
    a1.set_ylim(-6, 0.6)
    a2.set_ylabel("grafos distintos entre\nlas super-sesiones")
    a2.set_xlabel("iteracion")
    a2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(RES / "comparacion_corridas.png")
    plt.close(fig)


def tabla_resumen(corridas):
    filas = ["| corrida | modo | semilla | iteraciones | tiempo | mejor score | en iter | grafo | grados mayores |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r, _ in sorted(corridas, key=lambda c: (c[0]["modo"], c[0]["seed"])):
        tipo = "arbol" if r["arbol"] else "no arbol"
        filas.append(f"| `{r['nombre']}` | {r['modo']} | {r['seed']} | {r['iters_corridas']} | {r['segundos']//60} min | "
                     f"**{r['mejor_score']:+.4f}** | {r['mejor_iter']} | {tipo}, $\\lambda_1$={r['lambda1']:.3f}, $\\mu$={r['mu']} | "
                     f"{r['grados_mayores']} |")
    texto = ("# Resumen de corridas (generado por graficar.py)\n\n"
             f"Objetivo: score > 0 (contraejemplo). Maximo global posible: +{OPTIMO:.4f} (grafo de Wagner).\n\n"
             + "\n".join(filas) + "\n")
    (RES / "resumen_corridas.md").write_text(texto, encoding="utf-8")


def figuras_verdad_terreno():
    vt = RES / "verdad_terreno_arboles19"
    if not (vt / "resumen.json").exists():
        return
    r = json.load(open(vt / "resumen.json"))
    hst = json.load(open(vt / "histograma.json"))
    bordes, cuentas = np.array(hst["bordes"]), np.array(hst["cuentas"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(bordes[:-1], np.maximum(cuentas, 0.5), width=np.diff(bordes) * 0.9, align="edge", color="#2a78d6")
    ax.set_yscale("log")
    ax.axvline(0, color=GRIS, lw=0.9, ls="--")
    ax.axvline(OPTIMO, color=GRIS, lw=0.9, ls=":")
    ax.text(0, ax.get_ylim()[1], " score 0: estrella $K_{1,18}$", va="top", fontsize=7, color=GRIS)
    ax.text(OPTIMO, ax.get_ylim()[1] / 8, f" +{OPTIMO:.4f}: grafo de Wagner\n ({r['arboles_score_positivo']} arboles con score > 0)",
            va="top", fontsize=7, color=GRIS)
    ax.set_xlabel(r"score  $\sqrt{n-1}+1-\lambda_1-\mu$")
    ax.set_ylabel("numero de arboles (log)")
    ax.set_title(f"Todos los arboles no isomorfos de {N} vertices ({r['total_arboles']:,}): distribucion del score")
    fig.tight_layout()
    fig.savefig(vt / "histograma_scores.png")
    plt.close(fig)

    top = json.load(open(vt / "top_arboles.json"))
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, d in zip(axs, top[:3]):
        G = nx.Graph()
        G.add_edges_from(d["aristas"])
        dibujar(G, ax, "#2a78d6", f"score {d['score']:+.4f}\n$\\lambda_1$={d['lambda1']:.4f}, $\\mu$={d['mu']}")
    fig.suptitle(f"Los 3 mejores arboles de {N} vertices (el primero es el contraejemplo de Wagner)")
    fig.tight_layout()
    fig.savefig(vt / "contraejemplo_wagner.png")
    plt.close(fig)


if __name__ == "__main__":
    corridas = []
    for carpeta in sorted(RES.iterdir()):
        if (carpeta / "historial.csv").exists() and (carpeta / "resumen.json").exists():
            corridas.append(figuras_corrida(carpeta))
            print("figuras de", carpeta.name)
    if corridas:
        figura_comparacion(corridas)
        tabla_resumen(corridas)
    figuras_verdad_terreno()
    print("listo")
