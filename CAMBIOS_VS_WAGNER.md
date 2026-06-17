# Cambios respecto a la implementación de Wagner

Registro explícito de las diferencias entre el código original de Wagner
(`wagner_original/`, Keras/TensorFlow) y mi reimplementación en PyTorch. La idea es
que se vea claramente qué se rehízo y por qué, más allá de traducir el algoritmo de
un framework a otro.

## Convención

Cada diferencia tiene un número. En el código la marco así:

```python
# >>> CAMBIO N vs Wagner: <titulo corto>
# <una o dos lineas explicando>
```

y acá abajo está la fila N con el detalle. Al agregar un cambio nuevo le pongo el
siguiente número y lo anoto en la tabla. Esta convención se usa en **todos** los
notebooks del repo (validación, Enfoque 1 y Enfoque 2).

## Tabla resumen

| # | Componente | Wagner (original) | Mi versión (PyTorch) | Por qué |
|---|---|---|---|---|
| 1 | Framework | Keras / TensorFlow 1.14 | PyTorch 2.x | El original solo corre en Python 3.6 + TF1; PyTorch corre en el entorno actual (3.11) y aprovecha GPU |
| 2 | Definición de la red | `Sequential` + `Dense` | `nn.Module` con init Xavier explícito | Mismo modelo 128-64-4-1; Xavier coincide con el `glorot_uniform` que Keras usa por defecto |
| 3 | Optimizador | SGD, lr 1e-4 | Adam lr 1e-2 + weight_decay 1e-4 + gradient clipping (norma 1.0) | Adam converge en menos iteraciones; weight decay y clipping estabilizan (Wagner no los usa) |
| 4 | Número de matching μ | `nx.max_weight_matching` | Edmonds–Blossom propio en Numba `@njit` | Mucho más rápido en el *hot path*; además **corregido** respecto al apéndice de la tesis, que subestimaba μ |
| 5 | Conexidad | `nx.is_connected` | BFS propio en `@njit` (`es_conexo`) | Evita networkx dentro de la recompensa para poder jitearla |
| 6 | Función de recompensa | Arma un `nx.Graph` y calcula fuera de JIT | `calcScore` completo en `@njit` (matriz de adyacencia directa, eigvalsh, BFS, matching) | La recompensa se evalúa miles de veces por iteración; jitearla es lo que más acelera |
| 7 | Generación de sesiones | `for` sesión por sesión (cuello de botella, lo dice el propio Wagner) | Vectorizada sobre las `n_sessions` (una `forward` por paso) | Rapidez |
| 8 | Paso de entrenamiento | `model.fit(elite, ...)` (Keras) | Bucle PyTorch explícito: `BCELoss`, `backward`, `clip_grad_norm_`, `step` | Control fino y explícito de la optimización |
| 9 | Persistencia | pickle + txt cada 20 iters | `.pkl` de súper-sesiones + `modelo.pt` (state_dict) cada 20; `contraejemplo.npy` al hallarlo | Formato nativo de PyTorch para retomar el entrenamiento |
| 10 | Interpretación de la palabra (nb 02) | una palabra = un grafo (largo $\binom{N}{2}$) | una palabra = un **par**: largo $2\binom{N}{2}$, primera mitad $G_1$, segunda mitad $G_2$ | Paso hacia Stanley, que compara dos grafos en vez de uno |
| 11 | Reconstrucción del grafo (nb 02) | rearma un grafo | **parte la palabra en 2** y rearma cada mitad como grafo, sin tocar el resto del algoritmo | Leer el par reusando el mismo recorrido triangular |
| 12 | Recompensa (nb 02) | $\sqrt{n-1}+1-\lambda_1-\mu$ sobre un grafo | función del **par** (provisional: conexos + pocas aristas; luego comparar $U_0$ de Stanley) | El objetivo ahora vive sobre dos grafos |

> Las filas **1–9** corresponden al notebook 01 (validación con Wagner, un solo
> grafo). De la **10** en adelante son del notebook 02 (par de grafos) y los que
> sigan, con numeración continua.

## Lo que NO cambia (a propósito)

Para que la comparación sea justa, mantengo idénticos: la conjetura, la recompensa
$\sqrt{n-1}+1-\lambda_1-\mu$, la codificación del estado ($2\binom{N}{2}$ = palabra
parcial + one-hot), el tamaño de la red (128-64-4-1), los percentiles élite/súper
(93/94) y `n_sessions = 1000`.

## Nota sobre el cambio #4

Es el más relevante para la tesis: el score "0.17 superior a Wagner" que se reportó
originalmente venía de un matching con bug que subestimaba μ. Con el blossom
corregido (validado contra NetworkX en cada corrida) la comparación es honesta.
