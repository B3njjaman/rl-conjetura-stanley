# GENERADO por exportar_notebook.py a partir de notebooks/01_validacion_wagner_conj21.ipynb
# No editar a mano: cualquier cambio va en el notebook y se vuelve a exportar.

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit
import math, os, pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# semilla para poder repetir corridas (igual hay varianza por el muestreo)
np.random.seed(0)
torch.manual_seed(0)
print("device:", device)

N = 19                       # vertices
MYN = N * (N - 1) // 2       # 171 aristas posibles = largo de la palabra
observation_space = 2 * MYN  # lo que ve la red
len_game = MYN               # una decision por arista

n_sessions = 1000            # sesiones nuevas por iteracion
PCT_ELITE = 93               # entrenamos con el top 7%
PCT_SUPER = 94               # el top 6% sobrevive a la iteracion siguiente

# >>> CAMBIO 3 vs Wagner: optimizador y regularizacion
# Wagner: SGD(lr=1e-4) y nada mas. Aqui Adam(1e-2) + weight_decay + grad clipping.
LR = 0.01
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# >>> CAMBIO 1 vs Wagner: framework Keras/TF -> PyTorch (todo el notebook)
# >>> CAMBIO 2 vs Wagner: red como nn.Module, init Xavier explicito (= default de Keras)
class ModeloGrafo(nn.Module):
    def __init__(self, obs, n1=128, n2=64, n3=4):
        super().__init__()
        self.fc1 = nn.Linear(obs, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.fc3 = nn.Linear(n2, n3)
        self.salida = nn.Linear(n3, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.salida(x))

# >>> CAMBIO 4 vs Wagner: matching propio en Numba (rapido) y CORREGIDO
# Wagner usa nx.max_weight_matching; la version del apendice de la tesis subestimaba mu.

@njit
def _lca(a, b, base, mate, padre, n):
    # ancestro comun mas bajo en el bosque alternante
    visto = np.zeros(n, dtype=np.uint8)
    while True:
        a = base[a]; visto[a] = 1
        if mate[a] == -1: break
        a = padre[mate[a]]
    while True:
        b = base[b]
        if visto[b]: return b
        b = padre[mate[b]]

@njit
def _marcar(v, b, hijo, base, mate, padre, blossom):
    # recorre el blossom marcandolo y -clave- fijando padres
    while base[v] != b:
        blossom[base[v]] = 1
        blossom[base[mate[v]]] = 1
        padre[v] = hijo                  # <- esto faltaba en la version del apendice
        hijo = mate[v]
        v = padre[mate[v]]

@njit
def _buscar(raiz, adj, mate, padre, n):
    # BFS buscando un camino aumentante desde la raiz
    for i in range(n): padre[i] = -1
    base = np.arange(n).astype(np.int64)
    usado = np.zeros(n, dtype=np.uint8); usado[raiz] = 1
    cola = np.empty(n + 5, dtype=np.int64); qs = 0; qe = 0
    cola[qe] = raiz; qe += 1
    while qs < qe:
        v = cola[qs]; qs += 1
        for to in range(n):
            if adj[v, to] == 0: continue
            if base[v] == base[to] or mate[v] == to: continue
            if to == raiz or (mate[to] != -1 and padre[mate[to]] != -1):
                cb = _lca(v, to, base, mate, padre, n)       # to es par -> blossom
                blossom = np.zeros(n, dtype=np.uint8)
                _marcar(v, cb, to, base, mate, padre, blossom)
                _marcar(to, cb, v, base, mate, padre, blossom)
                for i in range(n):
                    if blossom[base[i]]:
                        base[i] = cb
                        if not usado[i]: usado[i] = 1; cola[qe] = i; qe += 1
            elif padre[to] == -1:
                padre[to] = v
                if mate[to] == -1:
                    return to                                # camino aumentante
                else:
                    if not usado[mate[to]]:
                        usado[mate[to]] = 1; cola[qe] = mate[to]; qe += 1
    return -1

@njit
def numero_matching(adj):
    n = adj.shape[0]
    mate = -np.ones(n, dtype=np.int64)
    padre = -np.ones(n, dtype=np.int64)
    for v in range(n):
        if mate[v] == -1:
            u = _buscar(v, adj, mate, padre, n)
            while u != -1:                                   # aumenta alternando
                pv = padre[u]; ppv = mate[pv]
                mate[u] = pv; mate[pv] = u
                u = ppv
    return np.sum(mate != -1) // 2

# >>> CAMBIO 5 vs Wagner: conexidad por BFS en Numba (Wagner: nx.is_connected)
@njit
def es_conexo(adj):
    n = adj.shape[0]
    vis = np.zeros(n, dtype=np.uint8)
    cola = np.empty(n, dtype=np.int64); cola[0] = 0; vis[0] = 1
    ini = 0; fin = 1
    while ini < fin:
        u = cola[ini]; ini += 1
        for v in range(n):
            if adj[u, v] == 1 and not vis[v]:
                vis[v] = 1; cola[fin] = v; fin += 1
    return np.sum(vis) == n

# >>> CAMBIO 6 vs Wagner: recompensa entera en @njit (Wagner arma un nx.Graph fuera de JIT)
@njit
def calcScore(palabra, N):
    # palabra 0-1 de largo MYN -> matriz de adyacencia (triangular superior)
    adj = np.zeros((N, N), dtype=np.int64)
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            if palabra[c] == 1:
                adj[i, j] = 1; adj[j, i] = 1
            c += 1
    if not es_conexo(adj):
        return -1000.0                       # la conjetura pide G conexo
    ev = np.linalg.eigvalsh(adj.astype(np.float64))
    lam1 = np.max(np.abs(ev))                # radio espectral
    mu = numero_matching(adj)
    return math.sqrt(N - 1) + 1 - lam1 - mu  # >0 seria contraejemplo

# >>> CAMBIO 7 vs Wagner: generacion de sesiones vectorizada (Wagner: for por sesion, su cuello de botella)
def generar_sesiones(modelo, n_ses):
    # estados[s, :, t] = lo que vio la red en la sesion s, paso t (sirve para entrenar)
    estados = np.zeros((n_ses, observation_space, len_game), dtype=np.float32)
    acciones = np.zeros((n_ses, len_game), dtype=np.int8)
    estado = np.zeros((n_ses, observation_space), dtype=np.float32)
    estado[:, MYN] = 1                        # one-hot en la 1ra arista
    for paso in range(len_game):
        with torch.no_grad():
            prob = modelo(torch.from_numpy(estado).to(device)).cpu().numpy().reshape(-1)
        acc = (np.random.rand(n_ses) < prob).astype(np.int8)   # Bernoulli(p)
        estados[:, :, paso] = estado
        acciones[:, paso] = acc
        estado[:, paso] = acc                 # fijo la decision en la palabra
        estado[:, MYN + paso] = 0             # apago el one-hot actual
        if paso + 1 < len_game:
            estado[:, MYN + paso + 1] = 1     # prendo el de la arista siguiente
    return estados, acciones

def recompensas(acciones):
    r = np.empty(len(acciones))
    for j in range(len(acciones)):
        r[j] = calcScore(acciones[j].astype(np.int64), N)
    return r

def entrenar(iteraciones, n_ses=n_sessions, log_cada=1, guardar_cada=20, carpeta="../datos"):
    os.makedirs(carpeta, exist_ok=True)
    modelo = ModeloGrafo(observation_space).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)  # CAMBIO 3
    bce = nn.BCELoss()

    sup_est = np.zeros((0, observation_space, len_game), dtype=np.float32)
    sup_acc = np.zeros((0, len_game), dtype=np.int8)
    sup_rec = np.zeros(0)
    hist_max, hist_top = [], []

    for it in range(iteraciones):
        est, acc = generar_sesiones(modelo, n_ses)
        rec = recompensas(acc)

        # juntar con las super antes de seleccionar
        EST = np.concatenate([est, sup_est])
        ACC = np.concatenate([acc, sup_acc])
        REC = np.concatenate([rec, sup_rec])

        m_elite = REC >= np.percentile(REC, PCT_ELITE)
        m_super = REC >= np.percentile(REC, PCT_SUPER)

        # >>> CAMBIO 8 vs Wagner: paso de entrenamiento explicito (Wagner: model.fit)
        # cada par (estado del paso -> accion tomada) de las sesiones elite
        X = EST[m_elite].transpose(0, 2, 1).reshape(-1, observation_space)
        y = ACC[m_elite].reshape(-1).astype(np.float32)
        opt.zero_grad()
        pred = modelo(torch.from_numpy(X).to(device))
        loss = bce(pred, torch.from_numpy(y).to(device).unsqueeze(1))
        loss.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
        opt.step()

        sup_est, sup_acc, sup_rec = EST[m_super], ACC[m_super], REC[m_super]
        hist_max.append(float(REC.max()))
        hist_top.append(float(np.sort(REC)[-100:].mean()))   # promedio del top-100

        if it % log_cada == 0:
            print(f"iter {it:4d} | mejor {REC.max():+.4f} | top100 {hist_top[-1]:+.4f} | loss {loss.item():.4f}")

        if REC.max() > 0:                       # contraejemplo!
            mejor = ACC[np.argmax(REC)]
            np.save(os.path.join(carpeta, "contraejemplo.npy"), mejor)
            print("CONTRAEJEMPLO encontrado, score =", REC.max())
            break

        # >>> CAMBIO 9 vs Wagner: guardo modelo.pt (state_dict de torch) + super-sesiones
        if it % guardar_cada == 0:
            with open(os.path.join(carpeta, f"super_iter_{it}.pkl"), "wb") as f:
                pickle.dump({"acciones": sup_acc, "recompensas": sup_rec}, f)
            torch.save(modelo.state_dict(), os.path.join(carpeta, "modelo.pt"))

    return modelo, hist_max, hist_top
