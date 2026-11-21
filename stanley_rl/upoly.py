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


def uk_arbol_combinaciones(padres, k):
    """Version anterior (recalcula las componentes en cada combinacion). Se deja para los tests."""
    padres = np.asarray(padres, dtype=np.int64)
    n = len(padres)
    base, claves, orden = base_arboles(n, k)
    return _uk_arbol(padres, n, k, claves, orden, len(base))


# ----------------------------------------------------------------------------------------
# Arboles: U_k incremental (CAMBIO 18)
# ----------------------------------------------------------------------------------------
# Los cortes se eligen en orden creciente de vertice hijo. Como padres[i] < i, al cortar c todavia
# no hay cortes debajo de c: la componente nueva mide sz[c] y la componente que lo contenia (la del
# ancestro cortado mas cercano, o la de la raiz) pierde sz[c]. Cada combinacion cuesta O(1) mas
# subir hasta el ancestro cortado, sin recalcular ni ordenar nada.
#
# Para no ordenar los tamanos, a cada multiconjunto {s_1, ..., s_m} se le asocia
#     S = r[s_1] + ... + r[s_m]  (mod 2^64),  r[s] = numeros aleatorios de 64 bits fijos,
# que se actualiza en O(1) en cada corte. Una tabla hash lleva S al indice de la particion en la
# base. Al construir la tabla se comprueba que S es inyectiva sobre la base, y toda combinacion da
# una particion de la base, asi que el vector resultante es EXACTO.
_MASK64 = (1 << 64) - 1


@lru_cache(maxsize=None)
def _azar_tamanos(n):
    rng = np.random.default_rng(20261121)
    return rng.integers(0, _MASK64, size=n + 1, dtype=np.uint64, endpoint=True)


@njit(cache=True)
def _sonda(S, bits):
    return np.int64((S * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(64 - bits))


@lru_cache(maxsize=None)
def tabla_arboles(n, k):
    """(base, r, claves_tabla, valores_tabla, bits) para el U_k incremental de arboles con n vertices."""
    base = particiones(n, max_partes=min(k + 1, n))
    r = _azar_tamanos(n)
    sumas = []
    for p in base:
        s = 0
        for x in p:
            s = (s + int(r[x])) & _MASK64
        sumas.append(s)
    if len(set(sumas)) != len(sumas):
        raise RuntimeError("S no es inyectiva sobre la base: cambiar la semilla de _azar_tamanos")
    bits = max(4, int(np.ceil(np.log2(4 * len(base)))))
    claves = np.zeros(1 << bits, dtype=np.uint64)
    valores = -np.ones(1 << bits, dtype=np.int64)
    for i, s in enumerate(sumas):
        j = _sonda(np.uint64(s), bits)
        while valores[j] >= 0:
            j = (j + 1) & ((1 << bits) - 1)
        claves[j] = np.uint64(s)
        valores[j] = i
    return base, r, claves, valores, bits


@njit(cache=True)
def _mezcla(S):
    z = S + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _preorden(padres, n):
    """Reetiqueta el arbol en preorden (DFS desde 0): los descendientes de v quedan en
    (v, v + sz[v]). Devuelve (padres en preorden, sz en preorden)."""
    cnt = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n):
        cnt[padres[i] + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    hijos = np.empty(max(n - 1, 1), dtype=np.int64)
    pos = cnt[:n].copy()
    for i in range(1, n):
        hijos[pos[padres[i]]] = i
        pos[padres[i]] += 1
    nuevo = np.empty(n, dtype=np.int64)
    pila = np.empty(n, dtype=np.int64)
    tope = 0
    pila[0] = 0
    tope = 1
    t = 0
    while tope > 0:
        tope -= 1
        v = pila[tope]
        nuevo[v] = t
        t += 1
        for j in range(cnt[v + 1] - 1, cnt[v] - 1, -1):   # hijos en orden inverso
            pila[tope] = hijos[j]
            tope += 1
    np_ = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        np_[nuevo[i]] = nuevo[padres[i]]
    sz = np.ones(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        sz[np_[i]] += sz[i]
    return np_, sz


@njit(cache=True)
def _indice_tabla(S, claves, valores, bits):
    mascara = (1 << bits) - 1
    j = _sonda(S, bits)
    while True:                                  # carga <= 1/4: siempre hay celdas vacias
        v = valores[j]
        if v < 0:                                # celda vacia (tambien tiene clave 0): S no esta
            raise ValueError("particion fuera de la base")
        if claves[j] == S:
            return v
        j = (j + 1) & mascara


@njit(cache=True)
def _uk_inc(padres, n, k, r, claves, valores, bits, dim, solo_hash):
    """Recorre todos los conjuntos de <= k cortes. Devuelve (vector, hash).

    solo_hash=True: no arma el vector (dim se ignora) y devuelve el hash de 64 bits
    H = sum_combinaciones mezcla(S) = sum_lambda v_lambda * mezcla(S(lambda)), que depende solo
    del vector U_k (dos arboles con el mismo U_k tienen el mismo H).
    """
    padres, sz = _preorden(padres, n)
    out = np.zeros(1 if solo_hash else dim, dtype=np.int64)
    H = np.uint64(0)
    comp = np.zeros(k + 2, dtype=np.int64)
    cortado = np.zeros(k + 2, dtype=np.int64)    # vertice cortado en cada nivel (crecientes)
    ancestro = np.zeros(k + 2, dtype=np.int64)
    inicio = np.zeros(k + 2, dtype=np.int64)
    comp[0] = n
    S = r[n]
    nivel = 0
    inicio[0] = 1
    while True:
        # registrar la combinacion actual
        if solo_hash:
            H += _mezcla(S)
        else:
            out[_indice_tabla(S, claves, valores, bits)] += 1
        # ultimo nivel: todas las extensiones con un corte mas, sin tocar el estado
        if nivel == k - 1:
            for c in range(inicio[nivel], n):
                ia = 0
                for L in range(nivel, 0, -1):            # ancestro cortado mas cercano
                    a = cortado[L]
                    if c < a + sz[a]:
                        ia = L
                        break
                A = comp[ia]
                S2 = S - r[A] + r[A - sz[c]] + r[sz[c]]
                if solo_hash:
                    H += _mezcla(S2)
                else:
                    out[_indice_tabla(S2, claves, valores, bits)] += 1
            inicio[nivel] = n
        # siguiente combinacion (DFS sobre conjuntos crecientes de cortes)
        while True:
            if nivel < k - 1 and inicio[nivel] < n:
                c = inicio[nivel]
                inicio[nivel] += 1
                ia = 0
                for L in range(nivel, 0, -1):
                    a = cortado[L]
                    if c < a + sz[a]:
                        ia = L
                        break
                viejo = comp[ia]
                S = S - r[viejo] + r[viejo - sz[c]] + r[sz[c]]
                comp[ia] = viejo - sz[c]
                nivel += 1
                comp[nivel] = sz[c]
                cortado[nivel] = c
                ancestro[nivel] = ia
                inicio[nivel] = c + 1
                break
            if nivel == 0:
                return out, H
            c = cortado[nivel]
            ia = ancestro[nivel]
            viejo = comp[ia]
            S = S - r[viejo] + r[viejo + sz[c]] - r[sz[c]]
            comp[ia] = viejo + sz[c]
            nivel -= 1


def _validar(P, k):
    """Numba no revisa limites: se valida antes de entrar. Devuelve (P, n, k normalizado)."""
    P = np.ascontiguousarray(P, dtype=np.int64)
    if P.ndim != 2 or P.shape[1] < 1:
        raise ValueError("se espera una matriz de padres con n >= 1 columnas")
    n = P.shape[1]
    if n > 1 and ((P[:, 1:] < 0) | (P[:, 1:] >= np.arange(1, n))).any():
        raise ValueError("cada arbol debe cumplir 0 <= padres[i] < i para i >= 1")
    if int(k) != k or k < 0:
        raise ValueError("k debe ser un entero >= 0")
    return P, n, min(int(k), n - 1)              # con mas de n - 1 cortes no cambia nada


def uk_arbol(padres, k):
    """Vector del U restringido de nivel k del arbol dado por padres (padres[i] < i)."""
    P, n, k = _validar(np.asarray(padres).reshape(1, -1), k)
    padres = P[0]
    base, r, claves, valores, bits = tabla_arboles(n, k)
    return _uk_inc(padres, n, k, r, claves, valores, bits, len(base), False)[0]


@njit(cache=True, parallel=True)
def _uk_lote(P, n, k, r, claves, valores, bits, dim):
    out = np.zeros((P.shape[0], dim), dtype=np.int64)
    for s in prange(P.shape[0]):
        out[s] = _uk_inc(P[s], n, k, r, claves, valores, bits, dim, False)[0]
    return out


def uk_lote(P, k):
    """U_k de muchos arboles a la vez (filas de P = arreglos de padres), en paralelo."""
    P, n, k = _validar(P, k)
    base, r, claves, valores, bits = tabla_arboles(n, k)
    return _uk_lote(P, n, k, r, claves, valores, bits, len(base))


@njit(cache=True, parallel=True)
def _uk_hash_lote(P, n, k, r, claves, valores, bits):
    out = np.empty(P.shape[0], dtype=np.uint64)
    for s in prange(P.shape[0]):
        out[s] = _uk_inc(P[s], n, k, r, claves, valores, bits, 1, True)[1]
    return out


def uk_hash_lote(P, k):
    """Hash de 64 bits del U_k de cada arbol (igual U_k => igual hash; distinto U_k => distinto hash
    con muy poca probabilidad; los pesos son fijos, asi que un choque seria determinista y no hay
    garantia de 2^-64). Sirve como FILTRO: los candidatos se confirman con el vector exacto. Las
    coordenadas del vector caben en int64 en el rango de uso (n <= 34, k <= 6)."""
    P, n, k = _validar(P, k)
    base, r, claves, valores, bits = tabla_arboles(n, k)
    return _uk_hash_lote(P, n, k, r, claves, valores, bits)


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
