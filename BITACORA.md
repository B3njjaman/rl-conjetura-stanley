# Bitacora de trabajo

Registro cronologico del avance de la tesis. Para el detalle tecnico de las
diferencias con el codigo de Wagner ver [`CAMBIOS_VS_WAGNER.md`](CAMBIOS_VS_WAGNER.md);
aca va el avance general (que se hizo, que falta).

## 2026-06

- [x] Estructura inicial del repositorio.
- [x] Copia del codigo original de Wagner como referencia.
- [x] Notebook 01: validacion del metodo sobre la conjetura 2.1 de Wagner
      (autovalores + matching) en PyTorch.
- [x] Notebook 02: duplicar el algoritmo para generar *pares* de grafos
      (primer paso hacia la comparacion de polinomios de Stanley).
- [ ] Enfoque 1: recompensa basada en el U0-polinomio de pares de grafos.
- [ ] Enfoque 2: codificacion de arboles (Prufer inversa) y U0 truncado.
- [ ] Redaccion del capitulo de metodologia.

## Pendientes / ideas

- Verificar el calculo del U0-polinomio contra casos chicos conocidos (n <= 6).
- Medir el costo de evaluar la recompensa por iteracion tras jitear con Numba.
