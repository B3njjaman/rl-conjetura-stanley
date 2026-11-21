"""Estado y salida de un kernel lanzado con kaggle/lanzar.py.

    python kaggle/bajar.py --lote humo --estado      # solo el estado (running / complete / error)
    python kaggle/bajar.py --lote humo               # baja la salida a resultados/kaggle/<lote>/
"""
import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
USUARIO = "benjacampos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", required=True)
    ap.add_argument("--estado", action="store_true")
    a = ap.parse_args()
    ref = f"{USUARIO}/tesis-stanley-{a.lote.replace('_', '-')}"  # Kaggle usa guiones en el slug
    if a.estado:
        subprocess.run([sys.executable, "-m", "kaggle.cli", "kernels", "status", ref])
        return
    destino = RAIZ / "resultados" / "kaggle" / a.lote
    destino.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "kaggle.cli", "kernels", "output", ref, "-p", str(destino), "-o"],
                   check=True)
    print("salida en", destino)


if __name__ == "__main__":
    main()
