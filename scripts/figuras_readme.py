"""Figuras del README (modo claro y oscuro), a partir de los resultados guardados en el repo.

    python scripts/figuras_readme.py          # escribe figuras/readme/<nombre>-<claro|oscuro>.png

Sin redes neuronales: solo lee CSV/JSON y calcula U_k con Numba.
Paleta: la de referencia del metodo dataviz (categorica en orden fijo, superficies opacas).
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
SALIDA = RAIZ / "figuras" / "readme"
VT = RAIZ / "resultados" / "verdad_terreno"
KG = RAIZ / "resultados" / "kaggle"

TEMAS = {
    "claro": dict(fondo="#fcfcfb", tinta="#0b0b0b", tinta2="#52514e", suave="#d9d8d3", gris="#a8a7a1",
                  serie=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "oscuro": dict(fondo="#1a1a19", tinta="#f0efec", tinta2="#c3c2b7", suave="#383835", gris="#6f6e69",
                   serie=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}


def estilo(t):
    plt.rcParams.update({
        "figure.facecolor": t["fondo"], "axes.facecolor": t["fondo"], "savefig.facecolor": t["fondo"],
        "text.color": t["tinta"], "axes.labelcolor": t["tinta2"], "axes.edgecolor": t["suave"],
        "xtick.color": t["tinta2"], "ytick.color": t["tinta2"], "axes.titlecolor": t["tinta"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": t["suave"],
        "grid.linewidth": 0.8, "legend.frameon": False, "legend.labelcolor": t["tinta2"],
    })


def guardar(fig, nombre, modo):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / f"{nombre}-{modo}.png", dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def sin_ejes(ax):
    ax.set_axis_off()
    ax.grid(False)


# ----------------------------------------------------------------------------------------
# 1. El U-polinomio de un arbol: P4 y K_{1,3}, los 8 subconjuntos de aristas
# ----------------------------------------------------------------------------------------
def _componentes(n, aristas):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(aristas)
    return sorted((sorted(c) for c in nx.connected_components(G)), key=len, reverse=True)


def fig_upolinomio(modo):
    t = TEMAS[modo]
    estilo(t)
    arboles = {
        "$P_4$ (camino)": ({0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}, [(0, 1), (1, 2), (2, 3)]),
        "$K_{1,3}$ (estrella)": ({0: (1.5, 0.9), 1: (0, 0), 2: (1.5, 0), 3: (3, 0)}, [(0, 1), (0, 2), (0, 3)]),
    }
    subconj = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    fig, axs = plt.subplots(2, 8, figsize=(15, 4.4))
    for fila, (nombre, (pos, E)) in enumerate(arboles.items()):
        for j, S in enumerate(subconj):
            ax = axs[fila, j]
            sin_ejes(ax)
            A = [E[i] for i in S]
            comps = _componentes(4, A)
            color = {}
            for ci, c in enumerate(comps):
                for v in c:
                    color[v] = t["serie"][ci] if len(c) > 1 else t["gris"]
            for i, (u, v) in enumerate(E):
                (x1, y1), (x2, y2) = pos[u], pos[v]
                if i in S:
                    ax.plot([x1, x2], [y1, y2], color=color[u], lw=3, solid_capstyle="round", zorder=1)
                else:
                    ax.plot([x1, x2], [y1, y2], color=t["suave"], lw=1.5, ls=(0, (2, 2)), zorder=0)
            for v, (x, y) in pos.items():
                ax.scatter([x], [y], s=90, color=color[v], edgecolors=t["fondo"], linewidths=2, zorder=2)
            lam = "(" + ",".join(str(len(c)) for c in comps) + ")"
            ax.text(1.5, -0.75, lam, ha="center", va="top", fontsize=12, color=t["tinta"])
            ax.set_xlim(-0.4, 3.4)
            ax.set_ylim(-1.3, 1.3)
            if j == 0:
                ax.text(-0.4, 1.25, nombre, ha="left", va="bottom", fontsize=13, fontweight="bold",
                        color=t["tinta"])
    fig.text(0.01, -0.06,
             "Cada A ⊆ E deja unas componentes (en color; las líneas punteadas son aristas fuera de A). "
             "Sus tamaños forman la partición λ(A).\n"
             "P₄:  U = x₄ + 2·x₃x₁ + x₂² + 3·x₂x₁² + x₁⁴        "
             "K₁,₃:  U = x₄ + 3·x₃x₁ + 3·x₂x₁² + x₁⁴        → distintos: el U distingue estos dos árboles.",
             fontsize=11.5, color=t["tinta2"], ha="left")
    guardar(fig, "u-polinomio", modo)


# ----------------------------------------------------------------------------------------
# 2. U_k y el algoritmo incremental: tamanos de subarbol y cortes
# ----------------------------------------------------------------------------------------
PADRES_EJ = [0, 0, 0, 1, 1, 2, 3, 3, 6, 6]


def _pos_capas(padres):
    """Dibujo por capas (raiz arriba), x segun el orden de las hojas."""
    n = len(padres)
    hijos = {v: [] for v in range(n)}
    for v in range(1, n):
        hijos[padres[v]].append(v)
    x, prof, sig = {}, {0: 0}, [0]

    def rec(v, d):
        prof[v] = d
        if not hijos[v]:
            x[v] = sig[0]
            sig[0] += 1
        else:
            for h in hijos[v]:
                rec(h, d + 1)
            x[v] = np.mean([x[h] for h in hijos[v]])
    rec(0, 0)
    return {v: (x[v], -prof[v]) for v in range(n)}, hijos


def fig_uk(modo):
    t = TEMAS[modo]
    estilo(t)
    p = PADRES_EJ
    n = len(p)
    pos, hijos = _pos_capas(p)
    sz = [1] * n
    for v in range(n - 1, 0, -1):
        sz[p[v]] += sz[v]
    casos = [("Tamaño de cada subárbol", []),
             ("Quitar 1 arista (k = 1)", [3]),
             ("Quitar 2 aristas (k = 2)", [3, 5])]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (titulo, cortes) in zip(axs, casos):
        sin_ejes(ax)
        E = [(p[v], v) for v in range(1, n) if v not in cortes]
        comps = _componentes(n, E)
        color = {}
        for ci, c in enumerate(comps):
            for v in c:
                color[v] = t["serie"][ci] if cortes else t["serie"][0]
        for v in range(1, n):
            (x1, y1), (x2, y2) = pos[p[v]], pos[v]
            if v in cortes:
                ax.plot([x1, x2], [y1, y2], color=t["gris"], lw=1.5, ls=(0, (2, 2)), zorder=0)
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, "✂", ha="center", va="center", fontsize=15,
                        color=t["tinta"], zorder=3)
            else:
                ax.plot([x1, x2], [y1, y2], color=color[v], lw=2.5, zorder=1)
        for v, (x, y) in pos.items():
            ax.scatter([x], [y], s=430, color=color[v], edgecolors=t["fondo"], linewidths=2, zorder=2)
            ax.text(x, y, str(sz[v]) if not cortes else "", ha="center", va="center", fontsize=10.5,
                    color="#ffffff", fontweight="bold", zorder=4)
        ax.set_title(titulo, pad=12)
        if cortes:
            lam = "(" + ",".join(str(len(c)) for c in comps) + ")"
            ax.text(np.mean([q[0] for q in pos.values()]), -4.85, f"partición  λ = {lam}", ha="center",
                    fontsize=12.5, color=t["tinta"])
        else:
            ax.text(np.mean([q[0] for q in pos.values()]), -4.85, "n = 10 vértices, 9 aristas", ha="center",
                    fontsize=12.5, color=t["tinta"])
        ax.set_ylim(-5.2, 0.6)
    fig.text(0.01, -0.08,
             "U_k suma x_λ sobre todos los cortes de a lo más k aristas: 1 + 9 + 36 = 46 cortes para k = 2.  "
             "El algoritmo incremental recorre los cortes en profundidad:\n"
             "cortar la arista sobre el vértice c crea una componente de tamaño sz[c] (el número del panel "
             "izquierdo) y le resta sz[c] a la componente que lo contenía. Así no recalcula nada.",
             fontsize=11.5, color=t["tinta2"], ha="left")
    guardar(fig, "uk-cortes", modo)


# ----------------------------------------------------------------------------------------
# 3. Hasta que n distingue cada U_k
# ----------------------------------------------------------------------------------------
def yz_tamano(m):
    """Tamano del par Y/Z con igual U_m (Prop. 14 de arXiv:2002.08186)."""
    return 6 * 2 ** (m // 2) - 2 if m % 2 == 0 else int(18 * 2 ** ((m - 3) / 2) - 2)


def fig_phi(modo):
    t = TEMAS[modo]
    estilo(t)
    ks = np.arange(1, 7)
    yz = [yz_tamano(k) for k in ks]
    medido = {1: 7, 2: 10, 3: 16, 4: 22}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(ks, 0, yz, color=t["serie"][0], alpha=0.10, lw=0)
    ax.plot(ks, yz, color=t["serie"][0], lw=2, marker="o", ms=9, mfc=t["fondo"], mew=2,
            label="Tamaño del par Y/Z (Conjetura 15: es el mínimo)")
    ax.plot(list(medido), list(medido.values()), ls="none", marker="o", ms=10, color=t["serie"][1],
            mec=t["fondo"], mew=2, label="Mínimo medido en este repo (enumeración completa)", zorder=3)
    ax.plot([5, 5], [25, 34], color=t["serie"][1], lw=6, solid_capstyle="round", alpha=0.55, zorder=2)
    ax.annotate("U₅: abierto entre 25 y 34\n(este repo: nada hasta n = 24)", xy=(5, 29.5), xytext=(3.35, 36),
                fontsize=10.5, color=t["tinta"], arrowprops=dict(arrowstyle="-", color=t["tinta2"], lw=1))
    for k, v in zip(ks, yz):
        ax.text(k + 0.12, v - 0.5, f"{v}", fontsize=10.5, color=t["tinta2"], va="top")
    ax.text(2.6, 3, "zona donde U_k ya distingue todos los árboles", fontsize=10.5, color=t["tinta2"])
    ax.set_xlabel("k (cuántas aristas se pueden quitar)")
    ax.set_ylabel("n más chico con dos árboles\ndistintos y el mismo U_k")
    ax.set_xticks(ks)
    ax.set_ylim(0, 50)
    ax.set_title("¿Hasta qué tamaño distingue cada U_k?")
    ax.legend(loc="upper left", fontsize=10)
    guardar(fig, "phi-k", modo)


# ----------------------------------------------------------------------------------------
# 4. Cuantos pares hay
# ----------------------------------------------------------------------------------------
def conteos():
    u3, u4 = {}, {}
    for n in range(16, 20):
        u3[n] = json.load(open(VT / f"arboles_n{n}_k3.json"))["pares_no_isomorfos"]
        u4[n] = json.load(open(VT / f"arboles_n{n}_k4.json"))["pares_no_isomorfos"]
    for n in range(20, 25):
        d = json.load(open(VT / f"arboles_grandes_n{n}.json"))["niveles"]
        u3[n] = d["3"]["pares"]
        u4[n] = d["4"]["pares"]
    return u3, u4


def fig_pares(modo):
    t = TEMAS[modo]
    estilo(t)
    u3, u4 = conteos()
    ns = sorted(u3)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ns, [u3[n] for n in ns], color=t["serie"][0], lw=2, marker="o", ms=8, mec=t["fondo"], mew=2)
    n4 = [n for n in ns if u4[n] > 0]
    ax.plot(n4, [u4[n] for n in n4], color=t["serie"][1], lw=2, marker="o", ms=8, mec=t["fondo"], mew=2)
    ax.set_yscale("log")
    ax.text(ns[-1] + 0.2, u3[ns[-1]], f"igual U₃\n{u3[ns[-1]]:,}".replace(",", "."), color=t["tinta"],
            va="center", fontsize=10.5)
    ax.text(ns[-1] + 0.2, u4[ns[-1]], f"igual U₄\n{u4[ns[-1]]}", color=t["tinta"], va="center", fontsize=10.5)
    ax.text(16, 2.2e3, "igual U₅: ninguno hasta n = 24", color=t["tinta2"], fontsize=10.5)
    ax.set_xlim(15.5, 25.5)
    ax.set_xticks(ns)
    ax.set_xlabel("n (vértices del árbol)")
    ax.set_ylabel("pares de árboles distintos\ncon el mismo U_k (escala log)")
    ax.set_title("Cuántos pares existen (enumeración completa)")
    guardar(fig, "pares-por-n", modo)


# ----------------------------------------------------------------------------------------
# 5. Que algoritmo le pega: H3 (k = 3)
# ----------------------------------------------------------------------------------------
def exito(lote, sufijo):
    out = {}
    for n in (16, 17, 18):
        dirs = sorted(glob.glob(str(KG / lote / "resultados" / "04_h3_arboles" / f"N{n}_k3_s1*{sufijo}")))
        dirs = [d for d in dirs if d.endswith(sufijo) or (sufijo == "" and not d.endswith("_control"))]
        ok = [json.load(open(Path(d) / "resumen.json"))["pares_distintos"] > 0 for d in dirs]
        out[n] = 100 * np.mean(ok)
    return out


def fig_metodos(modo):
    t = TEMAS[modo]
    estilo(t)
    aprende = exito("h3_arboles", "")
    control = exito("h3_control", "_control")
    azar = json.load(open(VT / "azar_arboles_k3.json"))
    mem = json.load(open(VT / "azar_con_memoria_k3.json"))
    memo = {int(n): 100 * np.mean([r["pares_encontrados"] > 0 for r in v]) for n, v in mem.items()}
    teo = {16: 20.0, 17: 0.5, 18: 6.5}
    series = [("RL de Wagner (modo tesis)", aprende), ("Control: red sin aprender (lr = 0)", control),
              ("Azar uniforme, de a pares (cálculo exacto)", teo), ("Azar con memoria (sin redes)", memo)]
    ns = [16, 17, 18]
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (nombre, d) in enumerate(series):
        xs = np.arange(len(ns)) + (i - 1.5) * w
        vals = [d[n] for n in ns]
        ax.bar(xs, vals, width=w - 0.03, color=t["serie"][i], label=nombre, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:g}%", ha="center", va="bottom", fontsize=9.5, color=t["tinta2"])
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"n = {n}\n({p} par{'es' if p > 1 else ''} posible{'s' if p > 1 else ''})"
                        for n, p in zip(ns, (1, 1, 11))])
    ax.set_ylim(0, 118)
    ax.set_ylabel("% de semillas que encontraron un par")
    ax.set_title("¿Qué algoritmo encuentra los pares con igual U₃?  (mismo presupuesto: 6 millones de árboles)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    ax.grid(axis="x", visible=False)
    _ = azar
    guardar(fig, "metodos-k3", modo)


# ----------------------------------------------------------------------------------------
# 6. El colapso de la poblacion (Enfoque 1, N = 7)
# ----------------------------------------------------------------------------------------
def leer(ruta, cols):
    filas = list(csv.DictReader(open(ruta)))
    return [np.array([float(f[c]) for f in filas]) for c in cols]


def fig_colapso(modo):
    t = TEMAS[modo]
    estilo(t)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    base = KG / "enfoque1" / "resultados" / "02_enfoque1"
    for s in range(3):
        for suf, ci, et in (("", 0, "aprende"), ("_control", 1, "control lr = 0")):
            it, pares, sup = leer(base / f"N7_s{s}{suf}" / "historial.csv",
                                  ["iter", "pares_distintos", "super_distintos"])
            lab = et if s == 0 else None
            axs[0].plot(it + 1, pares, color=t["serie"][ci], lw=2, alpha=0.9, label=lab)
            axs[1].plot(it + 1, sup, color=t["serie"][ci], lw=2, alpha=0.9, label=lab)
    axs[0].axhline(106, color=t["tinta2"], lw=1, ls=(0, (3, 3)))
    axs[0].text(1.2, 109, "existen 106 pares", color=t["tinta2"], fontsize=10)
    axs[0].set_ylim(0, 120)
    axs[0].set_title("Pares distintos encontrados (grafos, N = 7)")
    axs[1].set_title("Súper sesiones distintas (diversidad)")
    axs[1].axvspan(170, 266, color=t["serie"][0], alpha=0.12, lw=0)
    axs[1].text(300, 40, "colapso:\niteraciones\n170 a 266", color=t["tinta"], fontsize=10)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("iteración (escala log)")
        ax.legend(loc="lower right" if ax is axs[0] else "center left", fontsize=10)
    fig.text(0.01, -0.05, "3 semillas por color. Al aprender, la población se llena de copias del mismo par "
             "(recompensa 1,0) y deja de explorar; la red congelada sigue sacando grafos distintos.",
             fontsize=11, color=t["tinta2"])
    guardar(fig, "colapso", modo)


# ----------------------------------------------------------------------------------------
# 7. Validacion: conjetura 2.1 de Wagner, N = 19
# ----------------------------------------------------------------------------------------
def fig_wagner(modo):
    t = TEMAS[modo]
    estilo(t)
    base = KG / "validacion" / "resultados" / "01_wagner_conj21"
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in range(3):
        it, mejor = leer(base / f"wagner_fit_seed{s}" / "historial.csv", ["iter", "mejor"])
        ax.plot(it + 1, mejor, color=t["serie"][0], lw=2, label="ajuste de Wagner (SGD, minilotes, tope)"
                if s == 0 else None)
        pts = [(int(m.group(1)), float(m.group(2))) for m in
               re.finditer(r"iter\s+(\d+) \| mejor ([-\d.]+)", (base / f"notebook_seed{s}" / "log.txt").read_text())]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(np.array(xs) + 1, ys, color=t["serie"][1], lw=2, marker="o", ms=5,
                    label="modo de la tesis (Adam, sin tope; muere por memoria)" if s == 0 else None)
    ax.axhline(0, color=t["tinta"], lw=1.2)
    ax.text(1.2, 0.15, "> 0: contraejemplo   (punteada: +0,0804, el máximo)", fontsize=10, color=t["tinta"])
    ax.axhline(0.0804, color=t["tinta2"], lw=1, ls=(0, (3, 3)))
    ax.text(1.2, -0.38, "−0,48: el «casi contraejemplo»", fontsize=10, color=t["tinta2"])
    ax.axhline(-0.4803, color=t["suave"], lw=1)
    ax.set_xscale("log")
    ax.set_ylim(-2.5, 0.6)
    ax.set_xlabel("iteración (escala log)")
    ax.set_ylabel("mejor puntaje  √18 + 1 − λ₁ − μ")
    ax.set_title("Validación: conjetura 2.1 con N = 19 (el máximo posible es +0,0804)")
    ax.legend(loc="center left", fontsize=10)
    guardar(fig, "wagner-n19", modo)


# ----------------------------------------------------------------------------------------
# 8. El par Y_{1,1} / Z_{1,1} (n = 22, igual U_4)
# ----------------------------------------------------------------------------------------
def fig_yz(modo):
    t = TEMAS[modo]
    estilo(t)
    ej = json.load(open(VT / "arboles_grandes_n22.json"))["ejemplos"]["4"][0]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for i, (ax, padres) in enumerate(zip(axs, ej)):
        sin_ejes(ax)
        T = nx.Graph([(padres[v], v) for v in range(1, len(padres))])
        pos = nx.kamada_kawai_layout(T)
        peso = {v: max(len(c) for c in nx.connected_components(T.subgraph(set(T) - {v}))) for v in T}
        cents = [v for v in T if peso[v] == min(peso.values())]
        c1, c2 = (cents + [None])[:2]
        nx.draw_networkx_edges(T, pos, ax=ax, edge_color=t["serie"][i], width=2.5)
        if c2 is not None:
            nx.draw_networkx_edges(T, pos, edgelist=[(c1, c2)], ax=ax, edge_color=t["tinta"], width=4)
        nx.draw_networkx_nodes(T, pos, ax=ax, node_color=t["serie"][i], node_size=150, edgecolors=t["fondo"],
                               linewidths=2)
        ax.set_title(["T₁", "T₂"][i], fontsize=15)
    fig.text(0.5, 0.0, "Dos árboles de 22 vértices, no isomorfos, con el mismo U₄ (y distinto U₅). "
             "Es el par más chico posible y coincide con el Y₁,₁ / Z₁,₁ de Aliste-Prieto, de Mier y Zamora.",
             ha="center", fontsize=11.5, color=t["tinta2"])
    guardar(fig, "par-u4-n22", modo)


if __name__ == "__main__":
    for modo in TEMAS:
        for f in (fig_upolinomio, fig_uk, fig_phi, fig_pares, fig_metodos, fig_colapso, fig_wagner, fig_yz):
            f(modo)
    print("listo:", sorted(p.name for p in SALIDA.glob("*.png")))
