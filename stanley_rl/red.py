"""La red de Wagner (128-64-4) en PyTorch y el paso de entrenamiento. Usa torch: correr en Kaggle.

Salida binaria (Enfoques de grafos): una neurona con sigmoide = P(poner la arista).
Salida categorica (Enfoque 2, arboles): n logits = a que vertice ya existente se conecta.
"""
import numpy as np
import torch
import torch.nn as nn


class Red(nn.Module):
    def __init__(self, obs, salidas=1, capas=(128, 64, 4)):
        super().__init__()
        dims = [obs, *capas]
        self.ocultas = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.salida = nn.Linear(dims[-1], salidas)
        self.binaria = salidas == 1
        for m in self.modules():                       # CAMBIO 2: Xavier (= glorot_uniform de Keras)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        for capa in self.ocultas:
            x = torch.relu(capa(x))
        x = self.salida(x)
        return torch.sigmoid(x) if self.binaria else x


def optimizador(modelo, cfg):
    if cfg.optim == "adam":
        return torch.optim.Adam(modelo.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=cfg.weight_decay)
    if cfg.optim == "sgd":
        return torch.optim.SGD(modelo.parameters(), lr=cfg.lr)
    raise ValueError(cfg.optim)


def paso_entrenamiento(modelo, opt, X, y, cfg, mascara=None):
    """Entrena con los pares (estado, accion) de las sesiones elite.

    cfg.minilote = None : UN paso con todo el lote (lo que hace el codigo de la tesis).
    cfg.minilote = 32   : una epoca por minilotes barajados (lo que hace model.fit de Wagner).
    mascara (solo categorica): True donde la accion es valida; las invalidas no cuentan.
    """
    dev = next(modelo.parameters()).device
    Xt = torch.from_numpy(X).to(dev)
    if modelo.binaria:
        yt = torch.from_numpy(y.astype(np.float32)).to(dev).unsqueeze(1)
        fn = nn.BCELoss()
    else:
        yt = torch.from_numpy(y.astype(np.int64)).to(dev)
        mt = torch.from_numpy(mascara).to(dev)
        fn = nn.CrossEntropyLoss()
    idx_lotes = [np.arange(len(X))] if not cfg.minilote else \
        np.array_split(np.random.permutation(len(X)), max(1, int(np.ceil(len(X) / cfg.minilote))))
    total = 0.0
    modelo.train()
    for idx in idx_lotes:
        it = torch.from_numpy(idx).to(dev)
        opt.zero_grad()
        out = modelo(Xt[it])
        if not modelo.binaria:
            out = out.masked_fill(~mt[it], -1e9)
        loss = fn(out, yt[it])
        loss.backward()
        if cfg.clip:
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip)
        opt.step()
        total += loss.item() * len(idx)
    return total / len(X)


def fn_probabilidades(modelo):
    """Convierte la red en una funcion numpy -> numpy (para generar sesiones sin gradiente)."""
    dev = next(modelo.parameters()).device

    def f(estado):
        modelo.eval()
        with torch.no_grad():
            out = modelo(torch.from_numpy(estado).to(dev)).cpu().numpy()
        return out.reshape(-1) if modelo.binaria else out
    return f
