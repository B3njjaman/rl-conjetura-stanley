"""Corre una instancia de los Enfoques 1 o 2 y deja todo en la carpeta de salida.

    python -m stanley_rl.correr e1 --N 5 --seed 0 --iters 300 --salida resultados/02_enfoque1/N5_s0
    python -m stanley_rl.correr e2 --N 24 --k 4 --seed 0 --iters 3000 --tiempo-max 36000 --salida ...

Los valores por defecto replican la tesis (ver Config de cada enfoque). Entrena una red: correr
en Kaggle o en el NLHPC, no en el PC.
"""
import argparse
import os
from dataclasses import fields

# un hilo por proceso: en Kaggle se corren varias instancias en paralelo
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# pares no isomorfos que existen (verdad de terreno, scripts/verdad_terreno/colisiones.py)
TOTAL_PARES_E1 = {5: 1, 6: 9, 7: 106}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("enfoque", choices=["e1", "e2"])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--tiempo-max", type=float, default=None)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--hilos", type=int, default=1)
    ap.add_argument("--total-pares", type=int, default=None)
    ap.add_argument("--parar-al-completar", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="otros campos de Config: lr=0.001 clip=none ...")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.hilos)
    if a.enfoque == "e1":
        from .enfoque1 import Config, entrenar
        cfg = Config(N=a.N, seed=a.seed)
        cfg.total_pares = a.total_pares or TOTAL_PARES_E1.get(a.N)
    else:
        from .enfoque2 import Config, entrenar
        cfg = Config(N=a.N, k=a.k, seed=a.seed)
        cfg.total_pares = a.total_pares
    if a.iters is not None:
        cfg.iters = a.iters
    cfg.tiempo_max = a.tiempo_max
    cfg.parar_al_completar = a.parar_al_completar
    tipos = {f.name: f.type for f in fields(cfg)}
    for kv in a.set:
        clave, valor = kv.split("=", 1)
        if valor.lower() == "none":
            v = None
        elif clave == "capas":
            v = tuple(int(x) for x in valor.split(","))
        elif tipos[clave] in (int, "int"):
            v = int(valor)
        elif tipos[clave] in (float, "float"):
            v = float(valor)
        else:
            v = valor
        setattr(cfg, clave, v)
    entrenar(cfg, a.salida, nombre=os.path.basename(a.salida.rstrip("/")))


if __name__ == "__main__":
    main()
