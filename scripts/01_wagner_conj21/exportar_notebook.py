# Exporta las celdas de codigo del notebook 01 a un modulo (nb01_core.py) para poder
# correr el mismo algoritmo como script largo, sin duplicar codigo a mano.
# Se saltan las celdas de "correr" y "graficar" y el chequeo lento contra networkx.
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
NB = RAIZ / "notebooks" / "01_validacion_wagner_conj21.ipynb"
SALIDA = Path(__file__).with_name("nb01_core.py")


def exportar():
    nb = json.load(open(NB, encoding="utf-8"))
    partes = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(s in src for s in ("ITERACIONES =", "plt.figure", "matching OK contra networkx")):
            continue
        partes.append(src)
    cab = ("# GENERADO por exportar_notebook.py a partir de notebooks/01_validacion_wagner_conj21.ipynb\n"
           "# No editar a mano: cualquier cambio va en el notebook y se vuelve a exportar.\n\n")
    SALIDA.write_text(cab + "\n\n".join(partes) + "\n", encoding="utf-8")
    print("exportado:", SALIDA, f"({len(partes)} celdas)")


if __name__ == "__main__":
    exportar()
