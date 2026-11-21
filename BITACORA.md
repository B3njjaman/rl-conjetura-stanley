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

Un hilo, árboles al azar; el «antes» varió entre 42 y 61 ms con n = 34 según la corrida.

**Validación externa** (`tests/test_aliste_yz.py`): se construyen los pares $Y_{k,l}$, $Z_{k,l}$ de
Aliste-Prieto, de Mier y Zamora (arXiv:2002.08186, ec. 14 y 16). Con el código, tienen el mismo $U_{k+l+2}$ y
distinto $U_{k+l+3}$ para (k, l) = (0,0), (1,0), (0,1), (2,0), (1,1), (2,1) y (1,2), como dice su Teorema 12;
(2,1) es el par conjeturado mínimo con $U_5$ (n = 34). Además, el único par que encontró la enumeración
completa con n = 10 (k = 2) y con n = 16 (k = 3) es exactamente su Y/Z.

**Revisión de Codex**: sin errores en árboles válidos; comparó con fuerza bruta los 7.661 casos de n ≤ 7, todo k.
Se corrigieron sus hallazgos con entradas inválidas: validar los padres antes de Numba, que no revisa límites;
normalizar k ≤ n − 1; exigir bloque ≥ 1 y kmax ≥ 3; que la tabla no confunda una celda vacía con S = 0; no
usar `assert` para la inyectividad. El hash de 64 bits no garantiza 2⁻⁶⁴ porque los pesos son fijos. Por eso se usa
solo como filtro: los pares se confirman con el vector exacto. Codex propone una DP sobre el árbol, que en su
prototipo fue 4 veces más rápida que la enumeración con n = 34 y 8 veces con n = 26 (pendiente).

**Línea base de H3 con padres uniformes**, exacta (`scripts/verdad_terreno/azar_arboles.py`; la fórmula se
comprueba contra los (n − 1)! árboles recursivos con n ≤ 8): la probabilidad por sesión de sacar el par es
7,5·10⁻⁸ con n = 16, 1,7·10⁻⁹ con n = 17 y 2,3·10⁻⁸ con n = 18. En 3·10⁶ sesiones, que son 3.000
iteraciones, eso da **20 %**, 0,5 % y 6,5 % de corridas con un par por azar. Con n = 16 el azar ya basta
para 2 de 10 semillas: la prueba que discrimina son n = 17 y 18. Como la red sin entrenar no es uniforme,
se lanza el control con lr = 0 y las mismas semillas (lote `h3_control`).

La enumeración de la verdad de terreno usa el hash directo y da los mismos
resultados con n = 16–19 (n = 19: de 13,7 s a 3,0 s con 2 hilos). Las corridas de Kaggle lanzadas antes
(commits 15e5b65, 4f54e9b8 y 653bd417) usan la versión anterior: el resultado es el mismo, solo cambia el
tiempo.

## 2026-11-21 (noche) — primeros resultados de Kaggle

**Verdad de terreno n = 20–24** (kernel `verdad-terreno`, 64 min en 4 núcleos, con la versión anterior del código):

| n | árboles | pares con igual $U_3$ | igual $U_4$ | igual $U_5$ |
|---|---|---|---|---|
| 20 | 823.065 | 96 | 0 | 0 |
| 21 | 2.144.505 | 111 | 0 | 0 |
| 22 | 5.623.756 | 695 | **1** | 0 |
| 23 | 14.828.074 | 856 | 1 | 0 |
| 24 | 39.299.897 | 4.765 | 6 | 0 |

El primer par con igual $U_4$ aparece con n = 22, el mínimo publicado (Smith, Smith y Tian; Aliste-Prieto, de Mier y
Zamora), y es su $Y_{1,1}$/$Z_{1,1}$ (`tests/test_aliste_yz.py`). No hay pares con igual $U_5$ hasta n = 24, lo que
es compatible con la Conjetura 15. Los choques de $U_3$ son pocos: 4.765 pares entre 39 millones de árboles. La
cascada exhaustiva, entonces, se juega casi entera en el primer filtro.

**Enfoque 1 contra el control con lr = 0** (kernel `enfoque1`): **el control encuentra más pares**. Con N = 7, al
final de las 5.000 iteraciones, va 58, 37 y 15 de 106 aprendiendo, contra 104, 101 y 77 sin aprender. La causa es
el colapso de la población: las súper sesiones distintas bajan de ~64 a 1 entre las iteraciones 170 y 266, y después
no aparece ningún par nuevo. H2 queda refutada y H5 confirmada (`docs/hipotesis.md`). Con N ≤ 7 el problema es
fácil para el azar: el primer par sale en la iteración 0 a 5 con o sin aprendizaje.

## 2026-11-21 (noche, 2) — H1, H3 y H4, azar con memoria y comparación con la presentación

Se bajaron los lotes `validacion`, `enfoque2`, `h3_arboles` y `h3_control` (todos completos; los
`modelo.pt` quedan fuera del repo por el `.gitignore`). El detalle está en `docs/hipotesis.md`.

- **H1 (N = 19):** el ajuste estilo Wagner llega a +0,0804 en la semilla 0 y a otro contraejemplo
  (+0,0155) en la 1; la 2 queda en −0,4803. El modo de la tesis queda en −0,4803, −1,13 y −1,21 y muere
  por memoria cerca de la iteración 400: la selección sin tope (cambio 13) deja crecer la élite a más de 20.000.
- **H3 (árboles, k = 3):** aprendiendo 2/10, 0/10 y 0/10 en n = 16, 17 y 18; el control lr = 0 da
  exactamente lo mismo. Refutada según el criterio fijado; con 0 contra 0 solo se descarta un efecto grande.
- **H4 (N = 12, k = 3):** llega a √6 en 3/3 semillas. Se verificó que √6 es el mínimo exacto enumerando
  los 551 árboles (3 pares a esa distancia).
- **N = 24:** 0 pares en 10 h con k = 3 y k = 4, aunque existen 4.765 y 6.
- **Azar con memoria** (`scripts/verdad_terreno/azar_con_memoria.py`): árboles sueltos con padres
  uniformes y choques de U₃ buscados en todo el historial. Con el mismo presupuesto (6·10⁶ árboles)
  encuentra todos los pares de n = 16, 17 y 18 en 10/10 semillas, en segundos. La limitación del
  Enfoque 2 está en la unidad de búsqueda: solo compara los dos árboles de cada sesión.

**Comparación con la presentación** (`scripts/comparar_presentacion.py`, salida en
`resultados/comparacion_presentacion.txt`; aristas leídas a ojo de las figuras):

| Lámina | Presentación | Réplica |
|---|---|---|
| Wagner, N = 19 | −0,48 («casi contraejemplo») | modo tesis: −0,4803, el mismo valor; ajuste Wagner: +0,0804 |
| Grafos, N = 5, 6 y 7 | 1 par por N | los pares de N = 6 y 7 de las láminas son reales y están en la verdad de terreno; la réplica encuentra 9/9 y hasta 58/106 (el control, hasta 104/106) |
| Árboles, N = 12, k = 3 | un par a distancia 16,6 | √6 = 2,449, el mínimo exacto |
| Árboles, N = 24, k = 3 y 4 | pares a distancia ~237 y ~670 (aprox.; el de k = 4 ni siquiera tiene las mismas hojas, 11 y 12) | 15,7 (control 7,6) y 19,2; ningún par |

El texto de la tesis (cap. 4, ~l. 1164) dice que la validación dio «≈0.17 vs ≈0.08»: no lo respaldan
ni las láminas (−0,48) ni la réplica, y con árboles el máximo es 0,0804.

## Pendientes / ideas

- Verificar el calculo del U0-polinomio contra casos chicos conocidos (n <= 6).
- Medir el costo de evaluar la recompensa por iteracion tras jitear con Numba.
- Kaggle: quedo un kernel privado `benjacampos/rl-stanley-nb01-wagner` que fallo
  porque la GPU asignada no es compatible con el torch de la imagen; si se
  quiere usar, subirlo con `--accelerator` (T4) o forzar CPU.
