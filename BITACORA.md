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

## 2026-09-05 — ¿el notebook 01 encuentra el contraejemplo de Wagner?

Objetivo del dia: revisar el notebook 01 y comprobar de verdad si la
reimplementacion en PyTorch llega al contraejemplo de Wagner (N = 19).

Paso a paso:

1. **Verdad de terreno.** El contraejemplo de Wagner son dos estrellas K_{1,8}
   cuyos centros se unen a traves de un vertice intermedio: lambda_1 = sqrt(10),
   mu = 2, score = sqrt(18) + 1 - sqrt(10) - 2 = +0.0804. La recompensa del
   notebook (`calcScore`) da exactamente ese valor sobre ese grafo, igual que
   NetworkX. Ademas, como el minimo de lambda_1 + mu sobre grafos conexos se
   alcanza en un arbol, se enumeraron **todos** los arboles no isomorfos de 19
   vertices (317 955) con `scripts/01_wagner_conj21/enumerar_arboles.py`:
   el maximo global es +0.0804 (el grafo de Wagner) y solo 2 arboles tienen
   score positivo. La estrella K_{1,18} da exactamente 0.
2. **Velocidad.** Una iteracion completa (1000 sesiones, N = 19) tarda ~1.2 s en
   CPU con la generacion vectorizada; 1000 iteraciones son ~20 min. No hace
   falta la nube para este ejercicio (Wagner hablaba de horas/dias porque su
   bucle era en Python puro).
3. **Scripts para corridas largas** en `scripts/01_wagner_conj21/`:
   `exportar_notebook.py` saca las celdas de codigo del notebook a
   `nb01_core.py` (el notebook sigue siendo la fuente de verdad), `variantes.py`
   tiene el ciclo de entrenamiento con dos modos (`notebook` = tal cual el
   notebook; `wagner_fit` = SGD 1e-4 por minilotes de 32 y seleccion con tope,
   como el `model.fit` de Wagner), `correr.py` corre una instancia y deja todo
   en `resultados/01_wagner_conj21/<modo>_seed<N>/`.
4. **Corridas lanzadas** (4 en paralelo, 2 hilos cada una): `notebook` semillas
   0 y 1 (1000 iteraciones) y `wagner_fit` semillas 0 y 1 (hasta 3000
   iteraciones o 1 h). Resultados y figuras en `resultados/01_wagner_conj21/`.
5. Repo pasado a **publico**.

Observacion preliminar (de una primera corrida que se cancelo): con el
entrenamiento tal cual esta en el notebook (Adam 1e-2, un solo paso full-batch
por iteracion) la politica colapsa en ~150 iteraciones (loss ~0.04, las 100
mejores sesiones son el mismo grafo) y queda atrapada en score -0.48 / -1.21,
lejos del +0.08. Es la misma cifra (-0.48) que la corrida "adam" honesta de la
tesis original. La hipotesis es que la culpa es de los cambios 3 y 8 de
`CAMBIOS_VS_WAGNER.md`; la variante `wagner_fit` esta para probarla.

## Pendientes / ideas

- Verificar el calculo del U0-polinomio contra casos chicos conocidos (n <= 6).
- Medir el costo de evaluar la recompensa por iteracion tras jitear con Numba.
- Kaggle: quedo un kernel privado `benjacampos/rl-stanley-nb01-wagner` que fallo
  porque la GPU asignada no es compatible con el torch de la imagen; si se
  quiere usar, subirlo con `--accelerator` (T4) o forzar CPU.
