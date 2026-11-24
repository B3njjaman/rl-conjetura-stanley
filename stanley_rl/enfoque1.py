"""Enfoque 1: pares de grafos generales con el mismo U-polinomio (y = 0).

Replica de Desktop/SQM/Tesis/codigos/GrafosCompletos_noisomorfos.ipynb (codigo original de la
tesis), con el mismo nucleo:
  * palabra binaria de largo 2*C(N,2): primera mitad G1, segunda mitad G2 (CAMBIOS 10-11);
  * estado = palabra parcial + one-hot de la posicion (largo 2*L), red 128-64-4-1 con sigmoide;
  * recompensa 1 / (1 + MSE(v(G1), v(G2))) y NEG_INF = -1e6 si son isomorfos o alguno no es conexo;
  * Adam lr 1e-3, weight decay 1e-4, BCE, UN paso con todo el lote elite por iteracion;
  * elite = percentil 93 sin tope; super = percentil 94 con el tope de Wagner, ordenadas por
    recompensa (como el notebook).

Aceleraciones (no cambian la recompensa):
>>> CAMBIO 14: U-polinomio por programacion dinamica en Numba, en paralelo y solo para los grafos
    distintos del lote (upoly.py).
>>> CAMBIO 15: el chequeo de isomorfismo se hace solo cuando v(G1) = v(G2). Si los vectores
    difieren los grafos no pueden ser isomorfos (U es invariante), asi que la recompensa es la misma.
    El chequeo es exacto: forma canonica = minimo sobre las N! permutaciones (N <= 8).
>>> CAMBIO 16: sesiones generadas en bloque (una pasada de la red por paso para las 1000) y los
    estados de entrenamiento se reconstruyen desde las acciones (no se guardan).
"""
import time
from dataclasses import dataclass, field, asdict

import numpy as np
from numba import njit

from .ce import Registro, seleccionar
from .upoly import adjmask_desde_palabra, es_conexo_mask, u_signed_lote

NEG_INF = -1e6


@dataclass
class Config:
    N: int = 5
    seed: int = 0
    iters: int = 300
    tiempo_max: float = None
    n_ses: int = 1000
    pct_elite: float = 93
    pct_super: float = 94
    optim: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip: float = None
    minilote: int = None
    seleccion: str = "tesis"
    norma: str = "mse"
    capas: tuple = (128, 64, 4)
    parar_al_completar: bool = False      # parar si se encontraron todos los pares que existen
    total_pares: int = None               # verdad de terreno (1, 9, 106 para N = 5, 6, 7)
    extra: dict = field(default_factory=dict)


def dimensiones(N):
    M = N * (N - 1) // 2
    L = 2 * M
    return M, L, 2 * L


# ----------------------------------------------------------------------------------------
# Decodificacion, conexidad e isomorfismo (Numba)
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def decodificar(acc, N, M):
    S = acc.shape[0]
    A1 = np.zeros((S, N), dtype=np.int64)
    A2 = np.zeros((S, N), dtype=np.int64)
    for s in range(S):
        A1[s] = adjmask_desde_palabra(acc[s, :M], N)
        A2[s] = adjmask_desde_palabra(acc[s, M:], N)
    return A1, A2


@njit(cache=True)
def ambos_conexos(A1, A2, N):
    S = A1.shape[0]
    out = np.zeros(S, dtype=np.bool_)
    for s in range(S):
        out[s] = es_conexo_mask(A1[s], N) and es_conexo_mask(A2[s], N)
    return out


@njit(cache=True)
def _codigo(adjm, perm, n):
    code = 0
    bit = 0
    for a in range(n):
        for b in range(a + 1, n):
            if (adjm[perm[a]] >> perm[b]) & 1:
                code |= 1 << bit
            bit += 1
    return code


@njit(cache=True)
def canon(adjm, n):
    """Forma canonica exacta: minimo del codigo de aristas sobre las n! permutaciones (Heap)."""
    perm = np.arange(n)
    c = np.zeros(n, dtype=np.int64)
    best = _codigo(adjm, perm, n)
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                t = perm[0]; perm[0] = perm[i]; perm[i] = t
            else:
                t = perm[c[i]]; perm[c[i]] = perm[i]; perm[i] = t
            code = _codigo(adjm, perm, n)
            if code < best:
                best = code
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return best


def aristas_de_codigo(code, n):
    out, bit = [], 0
    for a in range(n):
        for b in range(a + 1, n):
            if (code >> bit) & 1:
                out.append((a, b))
            bit += 1
    return out


def _u_unicos(A, N):
    """U de cada fila de A, calculando una sola vez cada grafo repetido."""
    U, inv = np.unique(A, axis=0, return_inverse=True)
    return u_signed_lote(U, N)[inv.reshape(-1)]


class Recompensa:
    """Recompensa del Enfoque 1 + registro de los pares no isomorfos encontrados."""

    def __init__(self, N, norma="mse"):
        self.N = N
        self.M = N * (N - 1) // 2
        self.norma = norma
        self._canon = {}
        self.encontrados = {}          # (canon1, canon2) -> primera iteracion

    def canon(self, adjm):
        key = adjm.tobytes()
        c = self._canon.get(key)
        if c is None:
            c = int(canon(adjm, self.N))
            self._canon[key] = c
        return c

    def __call__(self, acc, it=0):
        N = self.N
        A1, A2 = decodificar(acc, N, self.M)
        rec = np.full(len(acc), NEG_INF)
        idx = np.nonzero(ambos_conexos(A1, A2, N))[0]
        if len(idx) == 0:
            return rec
        d = (_u_unicos(A1[idx], N) - _u_unicos(A2[idx], N)).astype(np.float64)
        dist = (d ** 2).mean(1) if self.norma == "mse" else np.sqrt((d ** 2).sum(1))
        rec[idx] = 1.0 / (1.0 + dist)
        for s in idx[dist == 0]:
            c1, c2 = self.canon(A1[s]), self.canon(A2[s])
            if c1 == c2:
                rec[s] = NEG_INF                          # isomorfos
            else:
                self.encontrados.setdefault((min(c1, c2), max(c1, c2)), it)
        return rec


# ----------------------------------------------------------------------------------------
# Sesiones y estados
# ----------------------------------------------------------------------------------------
def generar(prob_fn, n_ses, L, rng):
    """Igual que generate_session del notebook, pero las 1000 sesiones avanzan juntas."""
    acc = np.zeros((n_ses, L), dtype=np.int8)
    estado = np.zeros((n_ses, 2 * L), dtype=np.float32)
    estado[:, L] = 1
    for t in range(L):
        p = prob_fn(estado)
        a = (rng.random(n_ses) < p).astype(np.int8)
        acc[:, t] = a
        estado[:, t] = a
        estado[:, L + t] = 0
        if t + 1 < L:
            estado[:, L + t + 1] = 1
    return acc


def estados_de(acc, L):
    """Pares (estado antes del paso t, accion del paso t) de cada sesion, como en el notebook."""
    S = len(acc)
    X = np.zeros((S, L, 2 * L), dtype=np.float32)
    antes = np.tri(L, L, -1, dtype=np.float32)            # antes[t, j] = 1 si j < t
    X[:, :, :L] = acc[:, None, :].astype(np.float32) * antes[None]
    X[:, np.arange(L), L + np.arange(L)] = 1
    return X.reshape(S * L, 2 * L), acc.reshape(-1)


# ----------------------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------------------
def entrenar(cfg, carpeta, nombre=None):
    import torch
    from .red import Red, fn_probabilidades, optimizador, paso_entrenamiento

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    M, L, obs = dimensiones(cfg.N)
    reg = Registro(carpeta, nombre or f"e1_N{cfg.N}_s{cfg.seed}")
    reg.log(f"config {asdict(cfg)}", siempre=True)
    modelo = Red(obs, 1, cfg.capas)
    opt = optimizador(modelo, cfg)
    recompensa = Recompensa(cfg.N, cfg.norma)
    sup_acc = np.zeros((0, L), dtype=np.int8)
    sup_rec = np.zeros(0)
    t0 = time.time()
    it = -1
    for it in range(cfg.iters):
        if cfg.tiempo_max and time.time() - t0 > cfg.tiempo_max:
            reg.log(f"tope de tiempo en iter {it}", siempre=True)
            break
        acc = generar(fn_probabilidades(modelo), cfg.n_ses, L, rng)
        rec = recompensa(acc, it)
        ACC = np.concatenate([acc, sup_acc])
        REC = np.concatenate([rec, sup_rec])
        m_el, m_su = seleccionar(REC, cfg, cfg.n_ses)
        X, y = estados_de(ACC[m_el], L)
        loss = paso_entrenamiento(modelo, opt, X, y, cfg)
        orden = np.argsort(-REC[m_su], kind="stable")      # el notebook ordena las super
        sup_acc, sup_rec = ACC[m_su][orden], REC[m_su][orden]
        reg.fila(iter=it, mejor=float(REC.max()), top100=float(np.sort(REC)[-100:].mean()), loss=float(loss),
                 n_elite=int(m_el.sum()), n_super=int(len(sup_rec)),
                 super_distintos=len({r.tobytes() for r in sup_acc}),
                 sesiones_r1=int((rec == 1.0).sum()), pares_distintos=len(recompensa.encontrados))
        if cfg.parar_al_completar and cfg.total_pares and len(recompensa.encontrados) >= cfg.total_pares:
            reg.log(f"encontrados los {cfg.total_pares} pares que existen (iter {it})", siempre=True)
            break

    pares = [dict(primera_iter=i, g1=aristas_de_codigo(a, cfg.N), g2=aristas_de_codigo(b, cfg.N))
             for (a, b), i in sorted(recompensa.encontrados.items(), key=lambda x: x[1])]
    reg.json("pares.json", pares)
    torch.save(modelo.state_dict(), f"{carpeta}/modelo.pt")
    primera = min(recompensa.encontrados.values()) if recompensa.encontrados else None
    resumen = dict(enfoque=1, N=cfg.N, seed=cfg.seed, iters_corridas=it + 1, pares_distintos=len(pares),
                   total_pares_que_existen=cfg.total_pares, iter_primer_par=primera,
                   mejor_final=float(REC.max()) if it >= 0 else None)
    reg.cerrar(resumen)
    return resumen
