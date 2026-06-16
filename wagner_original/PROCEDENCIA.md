# Procedencia del código original de Wagner

Esta carpeta es una **copia sin modificar** del repositorio de Adam Zsolt Wagner,
incluida aquí como **referencia** para comparar contra mi reimplementación en
PyTorch.

- **Repositorio original:** https://github.com/zawagner22/cross-entropy-for-combinatorics
- **Commit clonado:** `a272a7ba8e3d2e418f1b036f34198d238ae4c242` (`a272a7b`, 2021-10-27)
- **Paper:** A. Z. Wagner, *Constructions in combinatorics via neural networks*, arXiv:2104.14516 (2021).

## Archivos

- `code_template.py` — plantilla general del método de Cross-Entropy.
- `demos/cem_binary_conj21.py` — **Conjetura 2.1** (el primer ejercicio): minimizar
  $\lambda_1(G) + \mu(G)$. Recompensa: `sqrt(N-1) + 1 - lambda1 - mu`, con `N = 19`.
  Si la recompensa es positiva, el grafo es un contraejemplo.
- `demos/cem_binary_conj23_with_numba.py` — Conjetura 2.3 (usa numba).

## Requisitos del código original (importante)

Wagner advierte que el código **solo corre bien** con:

- Python 3.6.3
- TensorFlow 1.14.0
- Keras 2.3.1

Con versiones más nuevas se rompe o se vuelve muy lento. Por eso para la
comparación uso este código como **especificación de referencia** (hiperparámetros,
codificación del estado y función de recompensa), y reimplemento el mismo algoritmo
en PyTorch para poder ejecutarlo en el entorno actual.

## Qué se compara

Mismo algoritmo (Deep Cross-Entropy), misma función de recompensa de la Conjetura 2.1,
mismos hiperparámetros (N=19, 1000 sesiones, percentiles 93/94, red 128-64-4-1).
El objetivo es verificar que la versión en PyTorch reproduce el comportamiento del
original de Wagner sobre el primer ejercicio.
