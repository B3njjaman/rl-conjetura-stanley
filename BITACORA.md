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

## 2026-11-21 — réplica acelerada de los Enfoques 1 y 2 y verdad de terreno

1. **Código del Enfoque 1 recuperado**: estaba en `Desktop/SQM/Tesis/codigos/GrafosCompletos_noisomorfos.ipynb`
   (con corridas de N = 5 y 6 en `Polinomiocromatico*`). Usa el U con signo $(-1)^{nulidad}$ (= $U_G(x, y=0)$,
   equivalente a $X_G$), recompensa $1/(1+\mathrm{MSE})$ y una sola penalización $-10^6$. El texto de la tesis dice
   otra cosa en los tres puntos (definición sin signo, norma 2, constantes $M$ y $D$): hay que corregir el texto.
   El del Enfoque 2 no aparece (carpetas `Uk_treesMH/*` vacías): se reconstruyó desde el cap. 3.
2. **Paquete `stanley_rl/`** con el mismo núcleo y cálculo acelerado (cambios 14–17). 27 tests sin redes
   (`python -m pytest tests`): el vector y la recompensa son idénticos a los del código original.
3. **Verdad de terreno** (`scripts/verdad_terreno/colisiones.py`, resultados en `resultados/verdad_terreno/`):
   - Grafos conexos con igual $U(x,0)$ (no isomorfos): **1 par** con N = 5 (el de Stanley 1995: corbata y
     diamante con cola, el mismo de `ResultadoN5.png`), **9** con N = 6 y **106** con N = 7.
   - Árboles con igual $U_3$: **ninguno hasta n = 15**; 1 par con n = 16, 1 con 17, 11 con 18, 12 con 19.
     Con $U_4$: ninguno hasta n = 19. **El experimento N = 12, k = 3 de la tesis no tenía ningún par que
     encontrar**: "no se encontraron pares" no dice nada del método ni de la conjetura.
4. Las corridas se mandan a Kaggle (`kaggle/lanzar.py`); en el PC no se entrena (se sobrecalienta).

## 2026-11-21 (tarde) — U_k incremental (cambio 18)

Para el objetivo estrella ($U_5$ con n = 26–33) el $U_k$ de la versión anterior costaba unos 40 ms por árbol
con n = 34: demasiado para 2.000 árboles por iteración. La versión nueva recorre los cortes en profundidad
(cada corte parte una sola componente) y traduce el multiconjunto de tamaños a su índice con una suma de
números aleatorios de 64 bits y una tabla hash, sin ordenar ni buscar. El vector es el mismo (50 tests).

| n | k | combinaciones | antes (ms) | ahora, vector (ms) | ahora, hash (ms) |
|---|---|---|---|---|---|
| 24 | 3 | 2.048 | 0,17 | 0,04 | 0,03 |
| 24 | 4 | 10.903 | 1,39 | 0,32 | 0,15 |
| 29 | 3 | 3.683 | 0,52 | 0,05 | 0,04 |
| 30 | 5 | 146.596 | 25,0 | 2,7 | 2,1 |
| 34 | 5 | 284.274 | 41,9 | 3,6 | 2,7 |

Un hilo, árboles al azar. La enumeración de la verdad de terreno usa el hash directo y da los mismos
resultados con n = 16–19 (n = 19: de 13,7 s a 3,0 s con 2 hilos). Las corridas de Kaggle lanzadas antes
(commits 15e5b65, 4f54e9b8 y 653bd417) usan la versión anterior: el resultado es el mismo, solo cambia el
tiempo.

## Pendientes / ideas

- Verificar el calculo del U0-polinomio contra casos chicos conocidos (n <= 6).
- Medir el costo de evaluar la recompensa por iteracion tras jitear con Numba.
- Kaggle: quedo un kernel privado `benjacampos/rl-stanley-nb01-wagner` que fallo
  porque la GPU asignada no es compatible con el torch de la imagen; si se
  quiere usar, subirlo con `--accelerator` (T4) o forzar CPU.
