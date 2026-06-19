# Notas: la conjetura de Stanley (1995) y el U0-polinomio

Notas de apoyo para la tesis. Resumen informal de los objetos que aparecen en el
problema; no pretende ser riguroso, sino fijar notacion.

## Polinomio simetrico cromatico

Para un grafo G = (V, E), el **polinomio simetrico cromatico** de Stanley es

    X_G(x_1, x_2, ...) = sum_kappa  prod_{v in V} x_{kappa(v)},

donde la suma recorre todas las coloraciones propias kappa : V -> {1, 2, ...}
(vertices adyacentes con color distinto). Es una funcion simetrica homogenea de
grado |V| que refina al polinomio cromatico usual.

## La conjetura

> **Conjetura (Stanley, 1995).** El polinomio X_T distingue arboles no isomorfos:
> si T_1, T_2 son arboles con X_{T_1} = X_{T_2}, entonces T_1 es isomorfo a T_2.

Equivalentemente, via el **U0-polinomio** de Noble-Welsh (1999): U_{0,T_1} = U_{0,T_2}
implica T_1 isomorfo a T_2. El U0-polinomio es mas comodo de calcular y es el que
se usa como senal de recompensa en este trabajo.

## Estado

- Verificada por computadora hasta n = 29 vertices.
- Abierta en general.

Por eso la busqueda de un contraejemplo (un par de arboles no isomorfos con el
mismo U0) es un problema de optimizacion con recompensa rara y discreta:
exactamente el tipo de problema que ataca el metodo de Wagner (2021).

## Referencias

Ver [`../referencias.bib`](../referencias.bib).
