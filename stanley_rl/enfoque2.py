"""Enfoque 2: pares de arboles con el mismo U-polinomio restringido U_k.

El codigo original de este enfoque no se encontro (las carpetas Uk_treesMH/* estan vacias), asi
que esta es una RECONSTRUCCION a partir del texto de la tesis (cap. 3, seccion Enfoque 2):
  * cada arbol se construye vertice a vertice: el vertice i (1 <= i <= n-1) elige a cual de los
    vertices 0..i-1 se conecta. La palabra del par tiene largo 2(n-1): T1 y luego T2;
  * la red elige con una softmax condicionada a los i padres posibles (CAMBIO 17: la red
    128-64-4 de Wagner con una capa de salida de n logits; los invalidos se enmascaran);
  * estado = padres ya elegidos en one-hot (2(n-1) x n) + one-hot de la posicion (2(n-1));
  * recompensa 1 / (1 + ||v_k(T1) - v_k(T2)||_2) y -M si T1 y T2 son isomorfos (no hace falta
    penalizar desconexion: la codificacion siempre da arboles);
  * Cross-Entropy categorico, Adam con los hiperparametros de la tabla de la tesis
    (lr 0.01, weight decay 1e-4, clipping 1.0), elite 93 / super 94.

Nota: 1/(1+MSE) y 1/(1+L2) ordenan las sesiones igual (una es funcion creciente de la otra), y el
metodo Cross-Entropy solo usa ese orden, asi que la eleccion de la norma no cambia la corrida.
"""
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache

import numpy as np

from .ce import Registro, seleccionar
from .upoly import uk_lote

PEN_ISO = -1e6


@dataclass
class Config:
    N: int = 12
    k: int = 3
    seed: int = 0
    iters: int = 1000
    tiempo_max: float = None
    n_ses: int = 1000
    pct_elite: float = 93
    pct_super: float = 94
    optim: str = "adam"
    lr: float = 1e-2
    weight_decay: float = 1e-4
    clip: float = 1.0
    minilote: int = None
    seleccion: str = "tesis"
    norma: str = "l2"
    capas: tuple = (128, 64, 4)
    parar_al_completar: bool = False
    total_pares: int = None
    extra: dict = field(default_factory=dict)


def dimensiones(n):
    L = 2 * (n - 1)
    return L, L * n + L


def vertice_del_paso(t, n):
    return t % (n - 1) + 1


# ----------------------------------------------------------------------------------------
# Forma canonica de arboles (AHU desde el/los centro/s)
# ----------------------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def canon_arbol(padres):
    n = len(padres)
    ady = [[] for _ in range(n)]
    for i in range(1, n):
        ady[i].append(padres[i])
        ady[padres[i]].append(i)
    grado = [len(a) for a in ady]
    hojas = [v for v in range(n) if grado[v] <= 1]
    restantes = n
    while restantes > 2:
        nuevas = []
        restantes -= len(hojas)
        for v in hojas:
            for u in ady[v]:
                grado[u] -= 1
                if grado[u] == 1:
                    nuevas.append(u)
            grado[v] = 0
        hojas = nuevas
    centros = hojas

    def codificar(raiz):
        orden, padre = [raiz], {raiz: -1}
        for v in orden:
            for u in ady[v]:
                if u != padre[v]:
                    padre[u] = v
                    orden.append(u)
        etiqueta = {}
        for v in reversed(orden):
            hijos = sorted(etiqueta[u] for u in ady[v] if u != padre[v])
            etiqueta[v] = "(" + "".join(hijos) + ")"
        return etiqueta[raiz]

    return min(codificar(c) for c in centros)


def padres_de(acc, n):
    S = len(acc)
    P1 = np.zeros((S, n), dtype=np.int64)
    P2 = np.zeros((S, n), dtype=np.int64)
    P1[:, 1:] = acc[:, :n - 1]
    P2[:, 1:] = acc[:, n - 1:]
    return P1, P2


def _uk_unicos(P, k):
    U, inv = np.unique(P, axis=0, return_inverse=True)
    return uk_lote(U, k)[inv.reshape(-1)]


class Recompensa:
    def __init__(self, n, k, norma="l2"):
        self.n, self.k, self.norma = n, k, norma
        self.encontrados = {}

    def __call__(self, acc, it=0):
        P1, P2 = padres_de(acc, self.n)
        d = (_uk_unicos(P1, self.k) - _uk_unicos(P2, self.k)).astype(np.float64)
        dist = np.sqrt((d ** 2).sum(1)) if self.norma == "l2" else (d ** 2).mean(1)
        rec = 1.0 / (1.0 + dist)
        for s in np.nonzero(dist == 0)[0]:
            c1 = canon_arbol(tuple(P1[s].tolist()))
            c2 = canon_arbol(tuple(P2[s].tolist()))
            if c1 == c2:
                rec[s] = PEN_ISO
            else:
                clave = (min(c1, c2), max(c1, c2))
                if clave not in self.encontrados:
                    self.encontrados[clave] = dict(iter=it, padres1=P1[s].tolist(), padres2=P2[s].tolist())
        return rec, dist


# ----------------------------------------------------------------------------------------
# Sesiones y estados
# ----------------------------------------------------------------------------------------
def mascara_validos(n):
    L, _ = dimensiones(n)
    i = np.array([vertice_del_paso(t, n) for t in range(L)])
    return np.arange(n)[None, :] < i[:, None]                    # (L, n)


def generar(logit_fn, n_ses, n, rng):
    L, obs = dimensiones(n)
    acc = np.zeros((n_ses, L), dtype=np.int64)
    estado = np.zeros((n_ses, obs), dtype=np.float32)
    estado[:, L * n] = 1
    filas = np.arange(n_ses)
    for t in range(L):
        i = vertice_del_paso(t, n)
        z = logit_fn(estado)[:, :i].astype(np.float64)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        u = rng.random((n_ses, 1))
        a = np.minimum((np.cumsum(p, 1) < u).sum(1), i - 1)       # muestreo categorico
        acc[:, t] = a
        estado[filas, t * n + a] = 1
        estado[:, L * n + t] = 0
        if t + 1 < L:
            estado[:, L * n + t + 1] = 1
    return acc


def estados_de(acc, n):
    S = len(acc)
    L, obs = dimensiones(n)
    onehot = np.zeros((S, L, n), dtype=np.float32)
    onehot[np.arange(S)[:, None], np.arange(L)[None, :], acc] = 1
    antes = np.tri(L, L, -1, dtype=np.float32)                   # antes[t, j] = 1 si j < t
    X = np.zeros((S, L, obs), dtype=np.float32)
    X[:, :, :L * n] = (antes[None, :, :, None] * onehot[:, None, :, :]).reshape(S, L, L * n)
    X[:, np.arange(L), L * n + np.arange(L)] = 1
    masc = np.broadcast_to(mascara_validos(n)[None], (S, L, n)).reshape(S * L, n)
    return X.reshape(S * L, obs), acc.reshape(-1), np.ascontiguousarray(masc)


# ----------------------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------------------
def entrenar(cfg, carpeta, nombre=None):
    import torch
    from .red import Red, fn_probabilidades, optimizador, paso_entrenamiento

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.N
    L, obs = dimensiones(n)
    reg = Registro(carpeta, nombre or f"e2_N{n}_k{cfg.k}_s{cfg.seed}")
    reg.log(f"config {asdict(cfg)}", siempre=True)
    modelo = Red(obs, n, cfg.capas)
    opt = optimizador(modelo, cfg)
    recompensa = Recompensa(n, cfg.k, cfg.norma)
    sup_acc = np.zeros((0, L), dtype=np.int64)
    sup_rec = np.zeros(0)
    t0 = time.time()
    mejor_dist = np.inf
    mejor_par = None
    it = -1
    for it in range(cfg.iters):
        if cfg.tiempo_max and time.time() - t0 > cfg.tiempo_max:
            reg.log(f"tope de tiempo en iter {it}", siempre=True)
            break
        acc = generar(fn_probabilidades(modelo), cfg.n_ses, n, rng)
        rec, dist = recompensa(acc, it)
        no_iso = rec > PEN_ISO
        if no_iso.any():
            j = int(np.argmin(np.where(no_iso, dist, np.inf)))
            if dist[j] < mejor_dist:
                mejor_dist = float(dist[j])
                P1, P2 = padres_de(acc[j:j + 1], n)
                mejor_par = dict(iter=it, dist=mejor_dist, padres1=P1[0].tolist(), padres2=P2[0].tolist())
        ACC = np.concatenate([acc, sup_acc])
        REC = np.concatenate([rec, sup_rec])
        m_el, m_su = seleccionar(REC, cfg, cfg.n_ses)
        X, y, masc = estados_de(ACC[m_el], n)
        loss = paso_entrenamiento(modelo, opt, X, y, cfg, mascara=masc)
        orden = np.argsort(-REC[m_su], kind="stable")
        sup_acc, sup_rec = ACC[m_su][orden], REC[m_su][orden]
        reg.fila(iter=it, mejor=float(REC.max()), top100=float(np.sort(REC)[-100:].mean()), loss=float(loss),
                 n_elite=int(m_el.sum()), n_super=int(len(sup_rec)),
                 super_distintos=len({r.tobytes() for r in sup_acc}),
                 iso=int((rec == PEN_ISO).sum()), mejor_dist_no_iso=mejor_dist,
                 pares_distintos=len(recompensa.encontrados))
        if cfg.parar_al_completar and cfg.total_pares and len(recompensa.encontrados) >= cfg.total_pares:
            reg.log(f"encontrados los {cfg.total_pares} pares (iter {it})", siempre=True)
            break

    reg.json("pares.json", list(recompensa.encontrados.values()))
    reg.json("mejor_par.json", mejor_par)
    torch.save(modelo.state_dict(), f"{carpeta}/modelo.pt")
    resumen = dict(enfoque=2, N=n, k=cfg.k, seed=cfg.seed, iters_corridas=it + 1,
                   pares_distintos=len(recompensa.encontrados), total_pares_que_existen=cfg.total_pares,
                   mejor_dist_no_iso=mejor_dist)
    reg.cerrar(resumen)
    return resumen
