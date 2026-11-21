"""Piezas comunes del metodo Cross-Entropy (Wagner 2021) que NO dependen de torch.

Seleccion de sesiones, registro de la corrida y utilidades. La red y el paso de entrenamiento
estan en red.py (torch) para que todo lo demas se pueda probar sin correr redes en el PC.
"""
import csv
import json
import os
import time

import numpy as np


# ----------------------------------------------------------------------------------------
# Seleccion de sesiones
# ----------------------------------------------------------------------------------------
def sel_percentil(REC, pct):
    """Todas las sesiones con recompensa >= percentil (sin tope).

    Es el select_elites del codigo de la tesis (notebook del Enfoque 1 y notebook 01).
    """
    return REC >= np.percentile(REC, pct)


def sel_wagner(REC, pct, n_ses, tipo):
    """Seleccion con tope de Wagner (code_template.py).

    Umbral = percentil pct. Las sesiones estrictamente por encima entran siempre; las que
    empatan con el umbral entran mientras quede cupo (cupo inicial n_ses * (100 - pct) / 100).
    Wagner descuenta el cupo distinto en sus dos funciones:
      tipo="elite": select_elites descuenta por cada sesion >= umbral (entre o no).
      tipo="super": select_super_sessions descuenta solo por las que entran.
    """
    thr = np.percentile(REC, pct)
    cupo = n_ses * (100.0 - pct) / 100.0
    mask = np.zeros(len(REC), dtype=bool)
    for i in range(len(REC)):
        if REC[i] >= thr - 1e-7:
            if cupo > 0 or REC[i] >= thr + 1e-7:
                mask[i] = True
                if tipo == "super":
                    cupo -= 1
            if tipo == "elite":
                cupo -= 1
    return mask


def seleccionar(REC, cfg, n_ses):
    """Mascaras elite y super segun la configuracion de la corrida.

    cfg.seleccion = "tesis"  : elite sin tope + super con tope (asi estaba el Enfoque 1).
    cfg.seleccion = "wagner" : elite y super con tope, como Wagner.
    cfg.seleccion = "sin_tope": ambas sin tope (notebook 01).
    """
    if cfg.seleccion == "tesis":
        return sel_percentil(REC, cfg.pct_elite), sel_wagner(REC, cfg.pct_super, n_ses, "super")
    if cfg.seleccion == "wagner":
        return sel_wagner(REC, cfg.pct_elite, n_ses, "elite"), sel_wagner(REC, cfg.pct_super, n_ses, "super")
    if cfg.seleccion == "sin_tope":
        return sel_percentil(REC, cfg.pct_elite), sel_percentil(REC, cfg.pct_super)
    raise ValueError(cfg.seleccion)


# ----------------------------------------------------------------------------------------
# Registro de la corrida
# ----------------------------------------------------------------------------------------
class Registro:
    """log.txt (una linea por iteracion), historial.csv y resumen.json en la carpeta de salida."""

    def __init__(self, carpeta, nombre, imprimir_cada=25):
        os.makedirs(carpeta, exist_ok=True)
        self.carpeta = carpeta
        self.nombre = nombre
        self.imprimir_cada = imprimir_cada
        self.flog = open(os.path.join(carpeta, "log.txt"), "w", buffering=1)
        self.filas = []
        self.t0 = time.time()

    def log(self, msg, siempre=False):
        self.flog.write(msg + "\n")
        if siempre:
            print(f"[{self.nombre}] {msg}", flush=True)

    def fila(self, **kw):
        kw["segundos"] = round(time.time() - self.t0, 1)
        self.filas.append(kw)
        partes = [f"{k} {v:+.4f}" if isinstance(v, float) and k != "segundos" else f"{k} {v}"
                  for k, v in kw.items()]
        self.log(" | ".join(partes), siempre=(kw.get("iter", 0) % self.imprimir_cada == 0))

    def json(self, nombre, obj):
        with open(os.path.join(self.carpeta, nombre), "w") as f:
            json.dump(obj, f, indent=1, default=_a_json)

    def cerrar(self, resumen):
        if self.filas:
            with open(os.path.join(self.carpeta, "historial.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.filas[0].keys()))
                w.writeheader()
                w.writerows(self.filas)
        resumen["segundos"] = round(time.time() - self.t0, 1)
        self.json("resumen.json", resumen)
        self.log("FIN " + json.dumps(resumen, default=_a_json), siempre=True)
        self.flog.close()


def _a_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
