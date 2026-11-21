"""Paquete de la tesis: Cross-Entropy (Wagner 2021) para buscar pares con el mismo U-polinomio.

Modulos:
    upoly     U-polinomio con signo (grafos) y restringido U_k (arboles), en Numba.
    ce        piezas comunes del metodo Cross-Entropy (seleccion elite/super, red, registro).
    enfoque1  pares de grafos generales (palabra binaria de largo 2*C(N,2)).
    enfoque2  pares de arboles (cada vertice elige padre; softmax enmascarada).
"""
