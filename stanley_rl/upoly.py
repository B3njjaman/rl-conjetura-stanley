"""U-polinomio de Noble-Welsh: version con signo (y = 0) para grafos generales y version
restringida U_k para arboles. Todo en enteros, con Numba.

Convenciones (las mismas del codigo original de la tesis, GrafosCompletos_noisomorfos.ipynb):

* Para A subconjunto de E, lambda(A) es la particion de n dada por los tamanos de las
  componentes conexas de (V, A).
* Grafos generales (Enfoque 1): el coeficiente de x_lambda es
      c_lambda = sum_{A : lambda(A) = lambda} (-1)^{|A| - r(A)},   r(A) = n - l(lambda),
  o sea U_G(x, y = 0) de Noble-Welsh. Por su Teorema 6.1, X_G = (-1)^n U_G(x_i = -p_i, y = 0),
  asi que dos grafos tienen el mismo vector si y solo si tienen la misma funcion simetrica
  cromatica X_G.
* Arboles (Enfoque 2): |A| = r(A) siempre, no hay signo, y el U restringido de nivel k suma solo
  sobre los A que se obtienen quitando a lo mas k aristas (particiones con a lo mas k + 1 partes).
  Es el U-polinomio restringido de Aliste-Prieto, de Mier y Zamora (2017).

>>> CAMBIO 14 vs tesis: el original recorria los 2^|E| subconjuntos de aristas con networkx.
    Aqui, para grafos generales, se usa una programacion dinamica sobre subconjuntos de
    vertices (O(3^n), no depende del numero de aristas) y para arboles se recorren solo las
    combinaciones de <= k aristas usando tamanos de subarbol. El vector resultante es identico
    (ver tests/test_upoly.py, que compara contra la funcion original).
"""
from functools import lru_cache

import numpy as np
from numba import njit, prange


# ----------------------------------------------------------------------------------------
# Particiones de enteros (la "base" del vector)
# ----------------------------------------------------------------------------------------
def particiones(n, max_partes=None):
    """Particiones de n en orden lexicografico decreciente: (n), (n-1,1), ..., (1,...,1).

    Es el mismo orden que generar_patrones_de_particion del codigo original.
    """
    def rec(k, maximo):
        if k == 0:
            return [[]]
        out = []
        for i in range(min(k, maximo), 0, -1):
            for sub in rec(k - i, i):
                out.append([i] + sub)
        return out

    todas = [tuple(p) for p in rec(n, n)]
    if max_partes is not None:
        todas = [p for p in todas if len(p) <= max_partes]
    return todas


def _clave(parts, base):
    """Codifica una particion (ordenada decreciente) como entero: sum parts[i] * base^i."""
    c = 0
    m = 1
    for p in parts:
        c += p * m
        m *= base
    return c


def tabla_claves(base_parts, n):
    """Claves enteras ordenadas de la base y la permutacion al orden de la base."""
    claves = np.array([_clave(p, n + 1) for p in base_parts], dtype=np.int64)
    orden = np.argsort(claves)
    return claves[orden], orden.astype(np.int64)


@njit(cache=True)
def _indice(clave, claves_ordenadas, orden):
    j = np.searchsorted(claves_ordenadas, clave)
    if j < len(claves_ordenadas) and claves_ordenadas[j] == clave:
        return orden[j]
    return -1


# ----------------------------------------------------------------------------------------
# Grafos generales: U(x, y=0) por programacion dinamica sobre subconjuntos de vertices
# ----------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def particiones_de_conjunto(n):
    """Todas las particiones de {0..n-1} como bloques (mascaras de bits) + indice de su tipo.

    Devuelve (bloques[P, n] con -1 de relleno, idx_tipo[P], base) donde base = particiones(n).
    Son B(n) particiones (Bell): 52, 203, 877, 4140, 21147 para n = 5..9.
    """
    base = particiones(n)
    claves, orden = tabla_claves(base, n)
    lista = []

    def rec(resto, bloques):
        if resto == 0:
            lista.append(list(bloques))
            return
        low = resto & -resto
        rest = resto ^ low
        sub = rest
        while True:                      # todos los subconjuntos de rest (incluido el vacio)
            bloques.append(low | sub)
            rec(rest & ~sub, bloques)
            bloques.pop()
            if sub == 0:
                break
            sub = (sub - 1) & rest

    rec((1 << n) - 1, [])
    P = len(lista)
    bloques = -np.ones((P, n), dtype=np.int64)
    idx = np.empty(P, dtype=np.int64)
    for i, bl in enumerate(lista):
        tam = sorted((bin(b).count("1") for b in bl), reverse=True)
        idx[i] = orden[np.searchsorted(claves, _clave(tam, n + 1))]
        for j, b in enumerate(bl):
            bloques[i, j] = b
    return bloques, idx, base


@njit(cache=True)
def _g_conexos(adjmask, n):
    """g[S] = sum sobre A en E(G[S]) con (S, A) conexo de (-1)^{|A|}.

    f[S] = sum_{A en E(G[S])} (-1)^{|A|} = 1 si S es independiente y 0 si no. Separando la
    componente que contiene al menor vertice de S:  f[S] = sum_{T, min(S) en T} g[T] f[S \\ T].
    """
    M = 1 << n
    indep = np.zeros(M, dtype=np.int64)
    g = np.zeros(M, dtype=np.int64)
    indep[0] = 1
    for S in range(1, M):
        low = S & -S
        v = 0
        while (1 << v) != low:
            v += 1
        rest = S ^ low
        indep[S] = 1 if (indep[rest] == 1 and (adjmask[v] & rest) == 0) else 0
        tot = indep[S]
        sub = (rest - 1) & rest if rest > 0 else 0
        # T = low | sub con sub subconjunto propio de rest  (T != S)
        if rest > 0:
            while True:
                T = low | sub
                tot -= g[T] * indep[S ^ T]
                if sub == 0:
                    break
                sub = (sub - 1) & rest
        g[S] = tot
    return g


@njit(cache=True)
def _u_signed_desde_g(g, bloques, idx_tipo, n, dim):
    out = np.zeros(dim, dtype=np.int64)
    P = bloques.shape[0]
    for p in range(P):
        prod = 1
        nb = 0
        for j in range(n):
            b = bloques[p, j]
            if b < 0:
                break
            prod *= g[b]
            nb += 1
            if prod == 0:
                break
        if prod != 0:
            # signo (-1)^{n - l}: c_lambda = (-1)^{n-l} sum_{pi tipo lambda} prod g(B)
            if (n - nb) % 2 == 1:
                prod = -prod
            out[idx_tipo[p]] += prod
    return out


@njit(cache=True)
def adjmask_desde_palabra(palabra, n):
    """Palabra 0-1 de largo C(n,2) (triangular superior por filas) -> mascaras de vecinos."""
    adjm = np.zeros(n, dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if palabra[c] == 1:
                adjm[i] |= 1 << j
                adjm[j] |= 1 << i
            c += 1
    return adjm


@njit(cache=True)
def es_conexo_mask(adjm, n):
    vis = 1
    frontera = 1
    while frontera:
        nuevo = 0
        f = frontera
        while f:
            low = f & -f
            v = 0
            while (1 << v) != low:
                v += 1
            nuevo |= adjm[v]
            f ^= low
        nuevo &= ~vis
        vis |= nuevo
        frontera = nuevo
    return vis == (1 << n) - 1


def u_signed(adjm, n):
    """Vector de U_G(x, y=0) en la base particiones(n). adjm = mascaras de vecinos (int64)."""
    bloques, idx, base = particiones_de_conjunto(n)
    g = _g_conexos(np.asarray(adjm, dtype=np.int64), n)
    return _u_signed_desde_g(g, bloques, idx, n, len(base))


def u_signed_nx(G):
    """Lo mismo para un networkx.Graph con nodos 0..n-1."""
    n = G.number_of_nodes()
    adjm = np.zeros(n, dtype=np.int64)
    for u, v in G.edges():
        adjm[u] |= 1 << v
        adjm[v] |= 1 << u
    return u_signed(adjm, n)


# ----------------------------------------------------------------------------------------
# Arboles: U restringido de nivel k (quitar a lo mas k aristas)
# ----------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def base_arboles(n, k):
    """Base del U_k de arboles con n vertices: particiones de n con a lo mas k + 1 partes."""
    assert (n + 1) ** (k + 1) < 2 ** 62, "clave entera de la particion se desborda"
    base = particiones(n, max_partes=k + 1)
    claves, orden = tabla_claves(base, n)
    return base, claves, orden


@njit(cache=True)
def _uk_arbol(padres, n, k, claves, orden, dim):
    """padres[i] < i para i = 1..n-1 (padres[0] se ignora). Las aristas son (i, padres[i])."""
    sz = np.ones(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        sz[padres[i]] += sz[i]
    out = np.zeros(dim, dtype=np.int64)
    out[_indice(np.int64(n), claves, orden)] += 1          # A = E: una sola componente
    m = n - 1
    comb = np.zeros(k, dtype=np.int64)
    tam = np.zeros(k + 1, dtype=np.int64)
    anc = np.zeros(k, dtype=np.int64)
    for j in range(1, min(k, m) + 1):
        for t in range(j):
            comb[t] = t + 1                               # vertices hijo 1..n-1 (= aristas)
        while True:
            # ancestro quitado mas cercano de cada vertice quitado (indices crecen hacia abajo)
            for a in range(j):
                anc[a] = -1
                d = comb[a]
                for b in range(a - 1, -1, -1):            # candidatos con indice menor
                    c = comb[b]
                    x = d
                    while x > c:
                        x = padres[x]
                    if x == c:
                        anc[a] = b
                        break
            raiz = n
            for a in range(j):
                tam[a] = sz[comb[a]]
            for a in range(j):
                if anc[a] == -1:
                    raiz -= sz[comb[a]]
                else:
                    tam[anc[a]] -= sz[comb[a]]
            tam[j] = raiz
            # ordenar decreciente (j + 1 <= k + 1 elementos: insercion)
            for a in range(1, j + 1):
                x = tam[a]
                b = a - 1
                while b >= 0 and tam[b] < x:
                    tam[b + 1] = tam[b]
                    b -= 1
                tam[b + 1] = x
            clave = np.int64(0)
            mult = np.int64(1)
            for a in range(j + 1):
                clave += tam[a] * mult
                mult *= n + 1
            out[_indice(clave, claves, orden)] += 1
            # siguiente combinacion de j elementos de {1..m}
            t = j - 1
            while t >= 0 and comb[t] == m - j + 1 + t:
                t -= 1
            if t < 0:
                break
            comb[t] += 1
            for s in range(t + 1, j):
                comb[s] = comb[s - 1] + 1
    return out


def uk_arbol(padres, k):
    """Vector del U restringido de nivel k del arbol dado por padres (padres[i] < i)."""
    padres = np.asarray(padres, dtype=np.int64)
    n = len(padres)
    base, claves, orden = base_arboles(n, k)
    return _uk_arbol(padres, n, k, claves, orden, len(base))


@njit(cache=True, parallel=True)
def _uk_lote(P, n, k, claves, orden, dim):
    out = np.zeros((P.shape[0], dim), dtype=np.int64)
    for s in prange(P.shape[0]):
        out[s] = _uk_arbol(P[s], n, k, claves, orden, dim)
    return out


def uk_lote(P, k):
    """U_k de muchos arboles a la vez (filas de P = arreglos de padres), en paralelo."""
    P = np.ascontiguousarray(P, dtype=np.int64)
    n = P.shape[1]
    base, claves, orden = base_arboles(n, k)
    return _uk_lote(P, n, k, claves, orden, len(base))


@njit(cache=True, parallel=True)
def _u_signed_lote(A, n, bloques, idx_tipo, dim):
    out = np.zeros((A.shape[0], dim), dtype=np.int64)
    for s in prange(A.shape[0]):
        g = _g_conexos(A[s], n)
        out[s] = _u_signed_desde_g(g, bloques, idx_tipo, n, dim)
    return out


def u_signed_lote(A, n):
    """U(x, y=0) de muchos grafos a la vez (filas de A = mascaras de vecinos), en paralelo."""
    bloques, idx, base = particiones_de_conjunto(n)
    return _u_signed_lote(np.ascontiguousarray(A, dtype=np.int64), n, bloques, idx, len(base))


def padres_desde_arbol_nx(T):
    """Reetiqueta un arbol de networkx (BFS desde 0) para que padres[i] < i."""
    import networkx as nx
    orden = list(nx.bfs_tree(T, 0).nodes()) if 0 in T else list(nx.bfs_tree(T, next(iter(T))).nodes())
    pos = {v: i for i, v in enumerate(orden)}
    padres = np.zeros(len(orden), dtype=np.int64)
    for v in orden[1:]:
        for u in T.neighbors(v):
            if pos[u] < pos[v]:
                padres[pos[v]] = pos[u]
                break
    return padres
