# Hipótesis de la tesis (versión 2.0, 21-11-2026)

Síntesis de dos revisiones independientes, una de Codex (GPT-6.2 Astra) y otra de Claude Fable, con el mismo brief y sin verse entre sí, más la verdad de terreno calculada en `resultados/verdad_terreno/`. Las citas marcadas ✔ se abrieron y comprobaron. El núcleo del método no cambia: es el Cross-Entropy de Wagner con la red 128-64-4.

## Qué cambió con la verdad de terreno

| Experimento de la tesis original | Lo que se sabe ahora | Qué se podía concluir |
|---|---|---|
| Validación de Wagner, N = 19 | Solo 2 de los 317.955 árboles tienen score > 0; el máximo es +0.0804. | Solo sirve como validación si una corrida da score > 0, verificado. |
| Enfoque 1, N = 5 | Existe **1** par, el de Stanley. Con aristas al azar sale en la iteración 0: se espera ~1,7 por cada 1.000 sesiones. | Nada sobre el aprendizaje si no hay un control sin aprendizaje. |
| Enfoque 1, N = 6 y 7 | Existen **9** y **106** pares. | Se puede medir el *recall*: la fracción de los pares que existen que el método encuentra. |
| Enfoque 2, N = 12, k = 3 | **No existe ningún par**. Con U₃ el primer par aparece en n = 16. | «No se encontraron pares» no dice nada ni del método ni de la conjetura. |
| Enfoque 2, N = 24, k = 3 y 4 | Con U₄ el par mínimo tiene n = 22 ✔ (Smith, Smith y Tian 2015, arXiv:1505.01889; Aliste-Prieto, de Mier y Zamora 2020, arXiv:2002.08186). La enumeración completa de n = 20 a 24 está corriendo en Kaggle. | Con el total de pares que existen, un resultado nulo pasa a ser medible. |
| «La conjetura podría ser verdadera» | n ≤ 29 ya está verificado (Heil y Ji). | Es una frase vacía. Se elimina. |

Resultados del 21-11 (detalle en cada hipótesis): H1 se reproduce con el ajuste de Wagner; H2 y H3 refutadas (el aprendizaje no supera al control lr = 0); H4 llega al óptimo √6; H5 confirmada. El azar con memoria encuentra todos los pares de n = 16–18 que ninguna versión del RL encontró.

## Hipótesis (las 5 aparecen en ambas revisiones; difiere la prioridad)

**H1. Validación.** El ajuste estilo Wagner (SGD 1e-4, minilotes de 32, selección con tope) llega a +0.0804 en N = 19, y el Adam de un paso de la tesis no.
- **Métrica:** tasa de éxito por semilla.
- **Se refuta si:** ninguna de las dos configuraciones llega a +0.0804 (entonces no se reproduce a Wagner y hay que decirlo), o si Adam también llega.
- **Resultado (21-11, lote `validacion`): se reproduce a Wagner, pero la comparación con Adam queda incompleta.**

  | | semilla 0 | semilla 1 | semilla 2 |
  |---|---|---|---|
  | Ajuste estilo Wagner | **+0,0804** (it. 5.967) | +0,0155 (otro contraejemplo, it. 5.433) | −0,4803 (12.698 it., 10 h) |
  | Modo de la tesis (Adam, selección sin tope) | −0,4803 | −1,1285 | −1,2068 |

  El modo de la tesis muere por memoria (código 137) cerca de la iteración 400 en las tres semillas: sin tope, la élite crece de 75 a más de 20.000 sesiones cuando hay empates arriba, y las súper sesiones distintas bajan a 9–11. El −0,48 es el mismo «casi contraejemplo» de la presentación (`Micontraejemplo2.png`). Para cerrar H1 falta correr el modo de la tesis con tope de memoria.

**H2. ¿Aprende el Enfoque 1?** Con N = 6 y 7, el Cross-Entropy descubre más pares distintos que la misma búsqueda sin aprendizaje, con el mismo número de sesiones.
- **Métrica:** recall contra sesiones, tanto la unión de las semillas como cada semilla por separado.
- **Se refuta si:** el recall del Cross-Entropy es menor o igual al del control.
- **Resultado (21-11, lote `enfoque1`): refutada.** Pares distintos encontrados al final, de los que existen:

  | | semilla 0 | semilla 1 | semilla 2 |
  |---|---|---|---|
  | N = 6, aprende (1.500 it) | 8/9 | 7/9 | 9/9 |
  | N = 6, control lr = 0 | 9/9 | 9/9 | 9/9 |
  | N = 7, aprende (5.000 it) | 58/106 | 37/106 | 15/106 |
  | N = 7, control lr = 0 | 104/106 | 101/106 | 77/106 |

  Hasta la iteración 100 van parejos (N = 7: 21, 25 y 4 pares aprendiendo; 15, 25 y 10 en el control). Después, la versión que aprende se estanca (ver H5). El primer par sale igual de rápido en ambas: iteración 0 a 5.

**H3. ¿Encuentra el Enfoque 2 pares que existen?** (prioridad 1 para Fable y para Claude; Codex prefería partir por H2, que es más barata). En árboles con k = 3 y n = 16, 17 y 18, el método encuentra un par muy por sobre lo que daría el azar.
- **Base (exacta):** con padres uniformes, la probabilidad por sesión es 7,5·10⁻⁸ (n = 16), 1,7·10⁻⁹ (n = 17) y 2,3·10⁻⁸ (n = 18). En 3.000 iteraciones eso da 20 %, 0,5 % y 6,5 % de corridas con éxito por azar (`resultados/verdad_terreno/azar_arboles_k3.json`). Con n = 16 el azar ya explica unas 2 de 10 semillas: **la evidencia está en n = 17 y 18**.
- **Control:** lote `h3_control`, igual pero con lr = 0, porque la red sin entrenar no es uniforme.
- **Se refuta si:** la tasa de éxito con aprendizaje no supera a la del control en n = 17 y 18.
- **Diseño:** 10 semillas fijadas antes de correr (101 a 110), 3.000 iteraciones, y la corrida para cuando encuentra todos los pares que existen.
- **Métricas:** tasa de éxito por semilla e iteraciones hasta el primer par.
- **Resultado (21-11, lotes `h3_arboles` y `h3_control`): refutada según el criterio fijado.** Semillas que encontraron el par:

  | | n = 16 (1 par) | n = 17 (1 par) | n = 18 (11 pares) |
  |---|---|---|---|
  | Aprende | 2/10 (it. 66 y 2.703) | 0/10 | 0/10 |
  | Control lr = 0 | 2/10 (it. 2.892 y 2.983) | 0/10 | 0/10 |
  | Azar uniforme por sesión (exacto) | 20 % | 0,5 % | 6,5 % |
  | **Azar con memoria** (ver abajo) | **10/10** | **10/10** | **10/10, los 11 pares** |

  Alcance: 0/10 contra 0/10 solo descarta un efecto grande del aprendizaje; no muestra que el RL sea peor que el azar por sesión. Lo que sí muestra es que **ninguno de los dos encuentra los pares de n = 17 y 18**, y que la mejor distancia final del control (1,4–2,0) es menor que la del que aprende (n = 17: casi siempre ~7,5; n = 18: hasta 11,5). El lote que aprende corrió con el código anterior al cambio 18 (mismo vector, otro tiempo).

  **Límites del resultado (segunda opinión, verificados en los datos):**
  - H3 corrió con la configuración por defecto de `enfoque2.Config`, que es el **modo de la tesis** (Adam lr = 1e-2, selección sin tope). Es el mismo modo que en H1 no llega a +0,0804. **El Enfoque 2 nunca corrió con el ajuste estilo Wagner**, así que H3 refuta esa configuración, no el método.
  - El que aprende colapsa: en 27 de 30 corridas termina con `n_elite` ≥ 1.000 (toda la población empatada en un par) y la mediana de la última mejora es la iteración 99. En el control, 0 de 30 y la mediana es la 750. Buscó con una fracción chica del presupuesto.
  - Con lr = 0 la red no cambia, así que el control genera sesiones i.i.d. de la red inicial (las súper sesiones solo pasan a la selección).
  - **Piso de √2:** en cada nivel j del U_k, los coeficientes suman C(n−1, j) en cualquier árbol. Por eso la diferencia de dos vectores distintos tiene al menos dos coordenadas no nulas, y la menor distancia no nula es √2. El control toca ese piso en 24 de 30 corridas sin encontrar el par: hay muchos «casi pares» que no son choques. Falta medir si una distancia chica ayuda en algo (medición sin redes propuesta en `BITACORA.md`).

**H4. Óptimo conocido sin colisión.** Con N = 12 y k = 3, la distancia mínima entre árboles no isomorfos es √6 ≈ 2,449, con 3 pares a esa distancia (verificado el 21-11 enumerando los 551 árboles).
- **Pregunta:** ¿llega el método a ese óptimo, o se queda atascado o colapsa a T₁ ≅ T₂?
- **Resultado (21-11, lote `enfoque2`): llega en 3/3 semillas** (2.000 iteraciones). El par de la presentación (`ArbolN12k3.png`) está a distancia 16,6.
- **N = 24 (10 h por corrida):** 0 pares con k = 3 (existen 4.765) y con k = 4 (existen 6). Mejor distancia: k = 3, 15,7 aprendiendo contra 7,6 en el control; k = 4, 19,2 y 28,5.

**Azar con memoria (control agregado el 21-11, sin redes).** Árboles sueltos con padres uniformes; los choques de U₃ se buscan en todo el historial, no dentro de cada sesión. Mismo presupuesto que H3 (6·10⁶ árboles), semillas 101–110 (`scripts/verdad_terreno/azar_con_memoria.py`, `resultados/verdad_terreno/azar_con_memoria_k3.json`). Encuentra todos los pares en las 30 corridas. Árboles hasta tenerlos todos: n = 16, de 2.681 a 75.058; n = 17, de 42.417 a 392.240; n = 18, de 102.249 a 3.478.942 (el primer par sale antes de los 24.000). Tarda de 1 a 25 s por semilla en el PC. Ojo: con n ≤ 18 hay solo 19.320, 48.629 y 123.867 árboles, así que 6·10⁶ muestras cubren casi todo el espacio; la comparación justa a escala es con la enumeración, no con el RL. Aun así muestra que **la unidad de búsqueda (un par por sesión) es lo que limita**: el azar con memoria detecta un choque apenas aparecen los dos árboles, en cualquier momento.

**H5. Sesgo de la codificación y bloqueo en el primer par.** Los pares que se encuentran son los que tienen más peso bajo la codificación por padres. Además, una corrida se queda en el primer par encontrado, porque su recompensa vale 1,0 y se queda en las súper sesiones.
- **Mitigación declarada:** degradar los pares ya encontrados, o medir el recall como la unión de varias semillas o reinicios. Codex propone una variante: comparar 4 reinicios de 50 iteraciones con una corrida de 200.
- **Resultado (21-11, Enfoque 1, N = 7): confirmada.** Las súper sesiones distintas pasan de unas 64 a **1** entre las iteraciones 170 y 266, porque la población colapsa en un solo par con recompensa 1,0. El último par nuevo aparece en las iteraciones 224, 132 y 167, y después no se encuentra ninguno más en las 4.800 restantes. Sin aprendizaje no hay colapso: el control sigue con unas 64 súper sesiones distintas y sigue encontrando pares.

## Objetivo estrella (decisión de Benja con sus profes)

**Buscar con RL pares de árboles con igual U₅ y n < 34.** La Conjetura 15 de Aliste-Prieto, de Mier y Zamora (2020) ✔ dice que el par más chico con igual U₅ tiene 34 vértices, y su Proposición 16(c) dice que esa conjetura implicaría la de Stanley.
- **Dos frentes que se complementan:** de n = 26 a 33 hay entre 2,8·10⁸ y 3·10¹¹ árboles.
  - Con el $U_k$ incremental, la enumeración exhaustiva en cascada ($U_3$ → $U_4$ → $U_5$) parece viable en el NLHPC hasta n ≈ 29–31; el número exacto sale de un piloto.
  - El RL busca donde ya no alcanza la enumeración, con un recall calibrado en los tramos donde sí alcanza.
- **Control positivo a escala:** el par $Y_{2,1}$, $Z_{2,1}$ de n = 34, que el código ya verifica (mismo $U_5$, distinto $U_6$).
- **Qué aporta:** un hallazgo refutaría la conjetura del profesor guía. Un resultado nulo quedaría calibrado por el recall medido en H3 (k = 3) y con k = 4 y n = 22 a 24.
- **Dónde correrlo:** es un buen uso del NLHPC.

## Riesgos metodológicos (y qué se reporta)

- **Penalización −10⁶ con selección por percentil.** Solo importa el orden. El riesgo está en la élite sin tope cuando hay empates arriba. Se reportan `n_elite`, `iso` y `super_distintos`.
- **El acantilado en T₁ ≅ T₂.** La penalización solo actúa a distancia 0, así que los mejores vecinos son pares casi isomorfos. Se reporta la fracción de pares isomorfos por iteración.
- **Codificación no uniforme.** El control usa la misma codificación, y el recall se reporta par por par.
- **MSE y L2 ordenan igual.** Elegir entre ellas no cambia la corrida.
- **Diferencias entre el texto y el código.** El texto de la tesis describe un U sin signo, norma 2 y constantes M y D, y nada de eso es lo que corrió. Hay que corregir el capítulo 3.

## Literatura para citar (comprobada ✔ o por comprobar)

- ✔ Smith, Smith y Tian (2015), *Symmetric Chromatic Polynomial of Trees*, arXiv:1505.01889: pares mínimos de «k-cortes»; con 3-cortes, n = 16 y un solo par.
- ✔ Aliste-Prieto, de Mier y Zamora (2020), *On the smallest trees with the same restricted U-polynomial and the rooted U-polynomial*, arXiv:2002.08186: Conjetura 15 y Proposición 16. No se encontró la revista; la DOI 10.1016/j.disc.2020.112255 está sin verificar.
- ✔ Ghebleh, Al-Yakoob, Kanso y Stevanović (2024), *Reinforcement learning for graph theory, I. Reimplementation of Wagner's approach*, arXiv:2403.18429.
- ✔ Angileri et al. (2024), *A Systematization of the Wagner Framework*, arXiv:2406.12667.
- Por comprobar: Charton, Ellenberg, Wagner y Williamson (2024), *PatternBoost*, arXiv:2411.00566; Mehrabian et al. (2023), arXiv:2311.03583; Roucairol y Cazenave (2022 o 2025), Monte Carlo para conjeturas espectrales.
