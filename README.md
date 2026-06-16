# RL para la búsqueda de contraejemplos a la conjetura de Stanley (1995)

Repositorio de trabajo de mi tesis de Licenciatura en Matemáticas (UNAB).

La **conjetura de Stanley (1995)** afirma que el polinomio simétrico cromático
distingue árboles no isomorfos: si dos árboles $T_1, T_2$ cumplen
$X_{T_1} = X_{T_2}$, entonces $T_1 \cong T_2$. De forma equivalente, en términos
del $U_0$-polinomio de Noble–Welsh (1999), si $U_{0,T_1} = U_{0,T_2}$ entonces
$T_1 \cong T_2$. La conjetura está verificada computacionalmente hasta $n = 29$
vértices y sigue abierta en general.

La idea de este trabajo es usar **aprendizaje reforzado profundo** (método de
*Cross-Entropy* + redes neuronales, al estilo de Wagner 2021) para buscar
automáticamente contraejemplos, es decir, pares de árboles no isomorfos con el
mismo polinomio.

## Plan de trabajo

1. **Validación del método** reproduciendo los resultados de Wagner (2021) sobre
   la conjetura de autovalores y *matching* ($\lambda_1 + \mu \ge \sqrt{n-1}+1$).
2. **Enfoque 1 — grafos generales:** pares de grafos no isomorfos con el mismo
   $U_0$-polinomio (codificación binaria de aristas, *Cross-Entropy* binario).
3. **Enfoque 2 — árboles:** pares de árboles con el mismo $U_0$-polinomio
   truncado (codificación de Prüfer inversa, *Cross-Entropy* categórico).

## Estructura

```
notebooks/   # cuadernos de trabajo (.ipynb)
figuras/     # gráficos generados
datos/       # checkpoints y sesiones (.pkl / .pt) — no versionado
```

## Entorno

```bash
pip install -r requirements.txt
```

Probado con Python 3.11, PyTorch 2.x (CPU/GPU) y Numba para acelerar el cálculo
de *matching* (Edmonds–Blossom), conexidad y polinomios.

## Referencias

- R. P. Stanley. *A symmetric function generalization of the chromatic polynomial of a graph.* Adv. Math. 111 (1995).
- S. D. Noble, D. J. A. Welsh. *A weighted graph polynomial from chromatic invariants of knots.* Ann. Inst. Fourier 49 (1999).
- A. Z. Wagner. *Constructions in combinatorics via neural networks.* arXiv:2104.14516 (2021).
