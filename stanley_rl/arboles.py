"""Enumeracion rapida de todos los arboles no isomorfos de n vertices (Numba).

Es el mismo algoritmo que networkx.nonisomorphic_trees (Wright, Richmond, Odlyzko y McKay 1986,
con el paso de Beyer-Hedetniemi para arboles con raiz), escrito sobre arreglos para no construir
un networkx.Graph por arbol. Cada arbol sale como arreglo de padres con padres[i] < i (el orden
de la sucesion de niveles es un preorden), que es lo que usa upoly.uk_arbol.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _siguiente_con_raiz(c, p):
    """Beyer-Hedetniemi en el lugar. p < 0: el ultimo indice con nivel > 1. False si se acabo."""
    n = len(c)
    if p < 0:
        p = n - 1
        while c[p] == 1:
            p -= 1
    if p == 0:
        return False
    q = p - 1
    while c[q] != c[p] - 1:
        q -= 1
    for i in range(p, n):
        c[i] = c[i - p + q]
    return True


@njit(cache=True)
def _segundo_uno(c):
    visto = False
    for i in range(len(c)):
        if c[i] == 1:
            if visto:
                return i
            visto = True
    return len(c)


@njit(cache=True)
def _siguiente_libre(c):
    """WROM en el lugar: deja en c el proximo arbol libre valido."""
    n = len(c)
    m = _segundo_uno(c)
    alto_izq = 0
    for i in range(1, m):
        if c[i] - 1 > alto_izq:
            alto_izq = c[i] - 1
    alto_resto = 0
    for i in range(m, n):
        if c[i] > alto_resto:
            alto_resto = c[i]
    largo_izq = m - 1
    largo_resto = n - m + 1
    valido = alto_resto >= alto_izq
    if valido and alto_resto == alto_izq:
        if largo_izq > largo_resto:
            valido = False
        elif largo_izq == largo_resto:
            # izquierda > resto (lexicografico); resto = [0] + c[m:]
            for j in range(largo_izq):
                a = c[1 + j] - 1
                b = 0 if j == 0 else c[m + j - 1]
                if a != b:
                    if a > b:
                        valido = False
                    break
    if valido:
        return
    p = largo_izq
    viejo = c[p]
    _siguiente_con_raiz(c, p)
    if viejo > 2:
        m2 = _segundo_uno(c)
        alto = 0
        for i in range(1, m2):
            if c[i] - 1 > alto:
                alto = c[i] - 1
        largo = alto + 1
        for j in range(largo):
            c[n - largo + j] = j + 1


@njit(cache=True)
def _niveles_a_padres(c, out):
    ultimo = np.zeros(len(c) + 1, dtype=np.int64)
    out[0] = 0
    ultimo[0] = 0
    for i in range(1, len(c)):
        out[i] = ultimo[c[i] - 1]
        ultimo[c[i]] = i


@njit(cache=True)
def _bloque(c, out):
    """Llena out con hasta len(out) arboles. Devuelve (cuantos, se_acabo)."""
    k = 0
    while k < out.shape[0]:
        _siguiente_libre(c)
        _niveles_a_padres(c, out[k])
        k += 1
        if not _siguiente_con_raiz(c, -1):
            return k, True
    return k, False


def arboles(n, bloque=500_000, dtype=np.int64):
    """Itera bloques (matriz de padres) con todos los arboles no isomorfos de n >= 3 vertices."""
    if n < 3:
        raise ValueError("n >= 3")
    c = np.array(list(range(n // 2 + 1)) + list(range(1, (n + 1) // 2)), dtype=np.int64)
    while True:
        out = np.empty((bloque, n), dtype=np.int64)
        k, fin = _bloque(c, out)
        yield out[:k].astype(dtype, copy=False)
        if fin:
            return
