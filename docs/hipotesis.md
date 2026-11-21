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

## Hipótesis (en las que coinciden ambas revisiones)

**H1. Validación.** El ajuste estilo Wagner (SGD 1e-4, minilotes de 32, selección con tope) llega a +0.0804 en N = 19, y el Adam de un paso de la tesis no.
- **Métrica:** tasa de éxito por semilla.
- **Se refuta si:** ninguna de las dos configuraciones llega a +0.0804 (entonces no se reproduce a Wagner y hay que decirlo), o si Adam también llega.
- **Estado:** lote `validacion` en Kaggle, 3 semillas por variante y hasta 10 h.

**H2. ¿Aprende el Enfoque 1?** Con N = 6 y 7, el Cross-Entropy descubre más pares distintos que la misma búsqueda sin aprendizaje, con el mismo número de sesiones.
- **Métrica:** recall contra sesiones, tanto la unión de las semillas como cada semilla por separado.
- **Se refuta si:** el recall del Cross-Entropy es menor o igual al del control.
- **Estado:** lote `enfoque1`, con controles de lr = 0.

**H3. ¿Encuentra el Enfoque 2 pares que existen?** (prioridad 1 según ambas revisiones). En árboles con k = 3 y n = 16, 17 y 18, el método encuentra un par muy por sobre lo que daría el azar.
- **Base:** con padres uniformes, la probabilidad por sesión es de 10⁻⁷ a 10⁻⁹ según Fable [por verificar con un cálculo exacto].
- **Diseño:** 10 semillas fijadas antes de correr (101 a 110), 3.000 iteraciones, y la corrida para cuando encuentra todos los pares que existen.
- **Métricas:** tasa de éxito por semilla e iteraciones hasta el primer par.
- **Estado:** lote `h3_arboles`, 30 corridas.

**H4. Óptimo conocido sin colisión.** Con N = 12 y k = 3, la distancia mínima entre árboles no isomorfos es √6 ≈ 2,449 (3 pares, según el cálculo de Fable [por verificar]).
- **Pregunta:** ¿llega el método a ese óptimo, o se queda atascado o colapsa a T₁ ≅ T₂?
- **Estado:** lote `enfoque2`, N = 12.

**H5. Sesgo de la codificación y bloqueo en el primer par.** Los pares que se encuentran son los que tienen más peso bajo la codificación por padres. Además, una corrida se queda en el primer par encontrado, porque su recompensa vale 1,0 y se queda en las súper sesiones.
- **Mitigación declarada:** degradar los pares ya encontrados, o medir el recall como la unión de varias semillas o reinicios. Codex propone una variante: comparar 4 reinicios de 50 iteraciones con una corrida de 200.

## Objetivo estrella (decisión de Benja con sus profes)

**Buscar con RL pares de árboles con igual U₅ y n < 34.** La Conjetura 15 de Aliste-Prieto, de Mier y Zamora (2020) ✔ dice que el par más chico con igual U₅ tiene 34 vértices, y su Proposición 16(c) dice que esa conjetura implicaría la de Stanley.
- **Por qué no se puede enumerar:** de n = 26 a 33 hay entre 10⁹ y 10¹¹ árboles. La búsqueda dirigida es justamente el caso de uso del método.
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
