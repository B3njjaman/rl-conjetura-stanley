# Alternativas al Cross-Entropy de Wagner para la conjetura de Stanley sobre árboles

Revisión de literatura, 21-11-2026. Pregunta: ¿qué otras formas hay de buscar dos árboles no isomorfos con igual U_k (o igual función simétrica cromática, CSF), y qué dice la literatura en contra de usar RL para esto?

**Leyenda.**
- ✔ = abrí la fuente (página abs de arXiv, DOI en Crossref, PDF del editor o del autor) y comprobé la afirmación concreta que se le atribuye.
- ? = no pude abrirla, o la afirmación la vi solo en otra fuente (se dice cuál).
- *Inferencia mía* = conclusión mía a partir de las fuentes. No está escrita en ellas.
- Las frases entre comillas en español son traducciones mías. Las que están en inglés son textuales.

**Tres truncaciones distintas: no mezclarlas.**
1. **U_k** (Aliste-Prieto, de Mier, Zamora): términos de grado ≤ k+1 de U(T), o sea, conjuntos de aristas a los que se les quitan ≤ k aristas. Es lo mismo que los «j-cuts» de Smith–Smith–Tian para j ≤ k. En el abstract de SST se habla de «k parts», que equivale a k−1 cortes.
2. **k-truncación de Heil–Ji**: solo particiones cuyas partes (componentes) tienen ≤ k vértices. La 2-truncación es el polinomio de matchings.
3. **Bosques con ≤ t aristas** (Awan, 2026): otra truncación más.

El resultado de 29 vértices de Heil–Ji es sobre su 3-truncación (Teorema 4.5), no sobre U_k. Como la 3-truncación se deduce de la CSF completa, la conjetura de Stanley queda verificada hasta 29 vértices.

---

## A. Estado del arte de la conjetura (1995–2026)

### A.1 Verificaciones computacionales

| Quién | Año | Hasta | Método | Verif. |
|---|---|---|---|---|
| L. Tan (comunicación personal) | — | 23 vértices | — | ? (solo citado por Heil–Ji) |
| I. Smith, Z. Smith, P. Tian, arXiv:1505.01889 | 2015 | abstract: 23; cuerpo: «5-cuts suficientes para distinguir todos los árboles hasta 24 vértices» y el preorden es total «para n ≤ 24» | árboles generados con `geng` (nauty), comparación lexicográfica de cortes ordenados, evaluación perezosa (1-cuts, luego 2-cuts solo en las clases empatadas, etc.) | ✔ |
| K. Russell, web de J. L. Martin | 2014 | 25 vértices | biblioteca C++ que genera árboles libres | ? (solo vía Heil–Ji, no abrí la web) |
| S. Heil, C. Ji, *Australas. J. Combin.* 75(2) (2019) 210–222, arXiv:1801.07363 | 2019 | **29 vértices** | algoritmo probabilístico; Teorema 4.5: la 3-truncación distingue todos los árboles ≤ 29 | ✔ |
| Posteriores | 2019–2026 | ninguna mayor encontrada | Orellana–Tomba (nov-2026) todavía cita 29 | ✔ (es una ausencia: no la encontré, no digo que no exista) |

**Heil y Ji 2019** ✔. *Australas. J. Combin.* 75(2) (2019) 210–222; arXiv:1801.07363.
- **Qué dicen.** Evalúan la CSF, escrita en la base de potencias, en residuos aleatorios módulo primos q (un homomorfismo φ_{q,C}). Si X_S **≠** X_T, el algoritmo lo prueba con probabilidad ≥ 1 − 2^{−k} en tiempo O(kn³). Lo verifiqué en ar5iv: es el Teorema 3.8 allí. El texto que extraje del PDF perdió el signo ≠.
  - En otras palabras, dos evaluaciones distintas certifican que los invariantes son distintos. Evaluaciones iguales solo sugieren igualdad.
- **Cómo verificaron.** Usaron la variante con C = (x, y, z, 0, 0, …), que corresponde a la 3-truncación:
  1. Calcularon φ_{q1,C1} para cada árbol.
  2. Agruparon por valor.
  3. Refinaron cada clase con φ_{q2,C2}, y así sucesivamente, hasta que todas las clases quedaron de un solo árbol.
- **Límite.** La barrera para pasar de 29 fue la memoria (Remark 4.6).
- **Código.** github.com/VietaFan/CSFTreeConjecture.
- **Pregunta que dejan abierta.** ¿La 3-truncación distingue todos los árboles?
- **Remark 4.1.** Cuentan que SST mostró que los cortes con |S| ≤ 5 bastan hasta 24 vértices.
- **Por qué sirve.** Es exactamente un **buscador de colisiones por huella (hash) y refinamiento de particiones**. Es la plantilla natural para buscar pares con igual U_5 por enumeración.

### A.2 Clases de árboles donde está probada, y la técnica de cada una

En la última columna, «excluir» significa sacar esa clase del espacio de búsqueda, porque ahí no hay contraejemplo a la CSF. Ojo: eso vale para la CSF completa. Un par con igual U_k sí puede caer en esas clases.

| Clase | Referencia | Técnica | Verif. | Para la búsqueda |
|---|---|---|---|---|
| Arañas, algunas orugas (y «squids» unicíclicos) | J. L. Martin, M. Morin, J. D. Wagner, *JCTA* 115 (2008) 237–253, arXiv:math/0609339 | S_T (polinomio de subárboles) = producto de Hall de X_T con una Φ fija; X_T determina las sucesiones de grados y de caminos | ✔ | excluir arañas; un par con igual CSF debe tener iguales sucesiones de grados y de caminos, que sirven como filtro barato |
| Orugas propias | J. Aliste-Prieto, J. Zamora, *Discrete Math.* 315–316 (2014) 158–164, DOI 10.1016/j.disc.2013.10.016 (arXiv:1208.2267) | correspondencia orugas ↔ composiciones de enteros | ✔ | queda subsumida por Loebl–Sereni |
| Todas las orugas | M. Loebl, J.-S. Sereni, *Ann. Inst. Henri Poincaré D* 6(3) (2019) 357–384, DOI 10.4171/aihpd/74 (arXiv:1405.4132) | el W-polinomio distingue árboles con pesos de una familia «buena» | ✔ | excluir orugas para la CSF; para U_k truncado no excluye nada, porque el resultado es sobre la CSF completa |
| Una clase nueva (sin nombre en el abstract) | Aliste-Prieto, de Mier, Zamora, *Discrete Math.* 340(6) (2017) 1435–1441 (arXiv:1509.09210) | subproducto del trabajo con Prouhet–Tarry–Escott | ✔ (abstract) | poco útil sin leer la definición (?) |
| Árboles con cierta descomposición de hojas | M. Gerling, arXiv:1702.06474 (2017) | particiones independientes con bloque de cardinal máximo | ✔ | clase técnica; poca poda práctica |
| Arañas generalizadas, caso n = 2 | J. Huryn, *Involve* 13 (2020) 109–116 (arXiv:1901.04034) | extensión del marco de arañas | ✔ | excluir |
| Núcleo y caminos colgantes | L. Crew, *Discrete Math.* 345(2) (2022) 112682 (arXiv:2106.04417) | X_T determina el tamaño del subárbol mínimo que contiene los vértices de grado ≥ 3 y el multiconjunto de largos de los caminos colgantes | ✔ | clave de hash previa: los dos árboles de un par deben coincidir en ella |
| Árboles propios de diámetro ≤ 5 | Aliste-Prieto, de Mier, R. Orellana, Zamora, *SIAM J. Discrete Math.* 37(3) (2023) 1881–1919 (arXiv:2202.11787) | grafos marcados, M-polinomio, algoritmo para la CSF en la base estrella | ✔ | excluir; además da un algoritmo eficiente para calcular la CSF en la base estrella |
| **Todos** los árboles de diámetro ≤ 5 | M. Gonzalez, R. Orellana, M. Tomba, arXiv:2404.06002 (v2 2025); FPSAC 2025, arXiv:2507.15986 | partición lexicográficamente mínima en la base estrella y su coeficiente; reconstrucción para diámetro < 6; el span de las CSF de árboles de n vértices tiene dimensión p(n) − n + 1 | ✔ | excluir diámetro ≤ 5; todo contraejemplo tiene diámetro ≥ 6 |
| q-orugas propias | G. Arunkumar, N. Narayanan, B. V. Raghavendra Rao, S. S. Sawant, *Discrete Math.* 347(11) (2024) 114162 (arXiv:2307.02234) | generaliza las orugas propias | ✔ | excluir |
| Exactamente dos vértices de grado ≥ 3 | Y. Wang, X. Yu, X.-D. Zhang, *Discrete Math.* 347(9) (2024) 114096 (arXiv:2308.03980) | derivación respecto de las sumas de potencias | ✔ | excluir; como los caminos (0 vértices de grado ≥ 3) y las arañas (1) también están resueltos, un contraejemplo necesita ≥ 3 vértices de grado ≥ 3 (inferencia mía) |
| Clase infinita de diámetro 6 | Z. Zeng, arXiv:2608.18179 (ago-2026) | núcleo con pesos; el diámetro 6 sin restricción sigue abierto | ✔ | excluir esa subclase; el diámetro 6 es la frontera abierta |
| Clase amplia de árboles propios | S. A. Awan, arXiv:2609.13464 (sep/oct-2026) | Φ_T = ∂X_T/∂p₁ en p₁ = 0; además prueba que contar bosques generadores con ≤ t aristas no distingue todos los árboles de k vértices si t < ⌊k/2⌋ | ✔ | excluir; el resultado sobre bosques es otra truncación, no U_k |

**Reducción más reciente.** R. Orellana, M. Tomba, *Proper implies improper for the chromatic symmetric function*, arXiv:2611.18940 (13-nov-2026) ✔ (HTML).
- **Teorema 4.11.** X_T = X_{T'} si y solo si X_{T(H_r)} = X_{T'(H_r)}.
- **Corolario 4.12.** Basta verificar la conjetura en los árboles «k-pesados»: árboles propios donde cada vértice que no es hoja tiene exactamente k−1 hojas.
- **Coeficientes redundantes.** En esa clase, los coeficientes con una parte igual a 1 se deducen de los demás.
- **Lectura para la búsqueda** *(inferencia mía)*. No achica el problema: un árbol k-pesado es su núcleo con una estrella colgada en cada vértice, así que buscar ahí es buscar en los núcleos. Lo que sí da es una comparación más barata, porque se pueden ignorar los coeficientes con partes 1.

**Invariantes más fuertes o variantes ya resueltas**
- **Versión con raíz, CSF.** N. Loehr, G. Warrington, arXiv:2206.05392 ✔: dos árboles con raíz son isomorfos (con raíz) si y solo si tienen el mismo polinomio cromático con raíz.
- **Versión con raíz, U.** arXiv:2002.08186, Teorema 9 ✔: el U-polinomio con raíz reconstruye el árbol con raíz.
- **Función cromática de Kneser.** Y. Nishimura, arXiv:2409.20478 ✔: K₂ distingue árboles. Es un invariante más fuerte que la CSF.
- **Versión cuasisimétrica.** J.-C. Aval, K. Djenabou, P. McNamara, *Algebraic Combin.* 6(3) (2023) 595–614 ✔: conjeturan que la función cromática cuasisimétrica distingue árboles dirigidos.
- **Álgebra de grupo.** A. M. Foley et al., arXiv:2112.09937 ✔: una «función madre» en el álgebra de grupo de S_n determina los árboles.
- **Núcleo de la CSF.** R. Penaguiao, arXiv:1803.08824 ✔: el núcleo de la CSF sobre grafos lo generan las relaciones modulares.
- **Por qué importa para la búsqueda.** Todo lo que está probado (orugas, arañas, diámetro ≤ 5, dos vértices de grado ≥ 3, etc.) se puede **excluir** del espacio de búsqueda.

### A.3 Invariantes relacionados y pares con igual U_k

**Definiciones de base**
- **Stanley (1995)** ✔. R. P. Stanley, *A symmetric function generalization of the chromatic polynomial of a graph*, *Adv. Math.* 111(1) (1995) 166–194, DOI 10.1006/aima.1995.1020.
- **Noble y Welsh (1999)** ✔. S. D. Noble, D. J. A. Welsh, *A weighted graph polynomial from chromatic invariants of knots*, *Ann. Inst. Fourier* 49(3) (1999) 1057–1087.
- **Relación entre ambos** ✔. Según arXiv:2002.08186, Remark 6: X(G) = (−1)^{|V|} U(G; x_i = −p_i, y = 0), y en árboles CSF y U son equivalentes.

**Existencia de pares (2017)** ✔. Aliste-Prieto, de Mier, Zamora, *Discrete Math.* 340(6) (2017) 1435–1441, arXiv:1509.09210.
- Para cada k existen árboles no isomorfos con igual U_k. La prueba usa soluciones del problema de Prouhet–Tarry–Escott (PTE).
- Los árboles no son explícitos. El paper de 2020 dice que encontrar soluciones PTE explícitas es difícil para k grande.

**Aliste-Prieto, de Mier, Zamora (2021): los pares Y/Z** ✔. *On the smallest trees with the same restricted U-polynomial and the rooted U-polynomial*, *Discrete Math.* 344(3) (2021) 112255 (DOI 10.1016/j.disc.2020.112255; arXiv:2002.08186, feb-2020). Lo leí completo. Lo que importa:
1. **U-polinomio con raíz U^r.** Tiene dos fórmulas de producto:
   - Lema 4: U^r(G ⊙ H) = z^{−1} U^r(G) U^r(H), donde ⊙ identifica las raíces.
   - Lema 5: U^r(G · H) = U^r(G) (U^r(H) + U(H)), donde · une las raíces con una arista y la raíz es la de G.
2. **Árboles A_k y B_k.**
   - A₀ es P₃ con raíz en el centro y B₀ es P₃ con raíz en una hoja.
   - A_k = A_{k−1} · B_{k−1} y B_k = B_{k−1} · A_{k−1}.
   - Son isomorfos sin raíz, pero U^r(A_k) − U^r(B_k) = Δ₀ P_k, con Δ₀ = x₁z² − x₂z y P_k = U(A₀)⋯U(A_{k−1}).
   - Todos los términos de P_k tienen grado ≥ k.
3. **Teorema 12.** Y_{k,l} = (A_k ⊙ A_l) · (B_k ⊙ B_l) y Z_{k,l} = (A_l ⊙ B_k) · (B_l ⊙ A_k) no son isomorfos, tienen **igual U_{k+l+2} y distinto U_{k+l+3}**.
4. **Tamaños.** |A_i| = 3·2^i y |Y_{k,l}| = 6(2^k + 2^l) − 2.
5. **Proposición 14 y Conjetura 15.** La Proposición 14 prueba Φ(m) ≤ 6·2^{m/2} − 2 si m es par, y Φ(m) ≤ 18·2^{(m−3)/2} − 2 si m es impar. La Conjetura 15 dice que vale la igualdad, es decir, que Y/Z son los pares más chicos con igual U_m.
   - El texto que extraje del PDF viene con la tipografía rota; reconstruí la fórmula desde |Y_{k,l}| y los (k, l) de la prueba de la Prop. 14.
   - Para m = 2, 3, 4 da 10, 16 y 22 vértices. El paper dice que coinciden con los mínimos que SST encontró por computador.
   - **El «34» para U_5 no está impreso en el paper.** Sale de la fórmula con m = 5: Y_{2,1}/Z_{2,1}, 6(4 + 2) − 2 = 34. Lo que está **probado** es que ese par de 34 vértices tiene igual U_5. Lo conjetural es que sea mínimo.
6. **Proposición 16.**
   - La conjetura de Stanley vale si y solo si Φ(m) → ∞.
   - La Conjetura 15 implica Stanley.
   - Para todo m, Stanley vale hasta Φ(m) − 1 vértices.

**Smith, Smith, Tian (2015)** ✔. arXiv:1505.01889. Lo leí completo. Es la **verdad de terreno** para cualquier buscador:

| invariante | primer n con pares | ejemplos |
|---|---|---|
| 1-cuts | 7 | 2 árboles |
| 2-cuts | 10 | n = 23: 1.643.204 árboles en 721.593 familias |
| 3-cuts | 16 | 1 familia en 16; 2 en 17; 11 familias (22 árboles) en 18; 92 en 20; 667 en 22; 769 (1.554 árboles) en 23 |
| 4-cuts | 22 | **exactamente un par** en 22 y uno en 23; el de 23 es el de 22 con la arista central subdividida |

- La densidad de árboles con 3-cuts repetidos es del orden de 10⁻⁴ (0,000105 en n = 23).
- Observan que los pares mínimos se obtienen **intercambiando ramas entre los dos centroides**.
- Conjeturan que subdividir la arista central de un par bicentroidal preserva los k-cuts.
- **Cómo leer las tablas** *(inferencia mía)*. Las leo como acumuladas, es decir, iguales en todos los j-cuts con j ≤ k, que es U_k. Eso sale del método perezoso (1-cuts, luego 2-cuts dentro de las clases empatadas, etc.) y de que 2002.08186 diga que sus mínimos coinciden con los de SST. Pero SST no lo dice explícitamente, y su propio ejemplo de 4 vértices tiene 2-cuts iguales con 1-cuts distintos. Antes de usar estas cifras como banco de prueba, conviene que el código de la tesis reproduzca los conteos de n = 16 a 18.
- **Cota inferior para U_5.** SST escribe que los 5-cuts bastaron «para distinguir todos los árboles hasta 24 vértices», pero también que la búsqueda fue exhaustiva «de 1 a 23 vértices». Según qué frase se tome, Φ(5) ≥ 24 o Φ(5) ≥ 25. Junto con el par Y_{2,1}/Z_{2,1}, la ventana abierta para el par mínimo con igual U_5 queda, siendo conservador, en **24 ≤ Φ(5) ≤ 34**. Es finita y enumerable.
- **Contraste con la tesis** *(inferencia mía)*. Si la tesis no encontró pares con U_3 en N = 12, eso calza con SST (el primero aparece en 16).
- **Contraste con la verdad de terreno del repo** (agregado el 21-11). Los mínimos coinciden: U_2 en n = 10, U_3 en 16 y U_4 en 22, con exactamente un par U_4 en 22 y otro en 23. Los conteos de U_3, en cambio, no calzan del todo:

  | n | SST | repo |
  |---|---|---|
  | 16 | 1 familia | 1 grupo |
  | 17 | 2 familias | 1 grupo |
  | 18 | 11 familias | 11 grupos |
  | 20 | 92 familias | 96 pares |
  | 22 | 667 familias | 695 pares |
  | 23 | 769 familias | 856 pares |

  Para n ≥ 20 el repo cuenta pares y SST familias, así que es esperable que difieran. La diferencia en n = 17 tiene que explicarse. La explicación más probable es que los «3-cuts» de SST no sean acumulados (no exigen igualdad en 1 y 2 cortes), pero **está pendiente de verificar**.
- **Cota inferior para U_5 con el repo.** La enumeración del repo revisó U_5 hasta n = 24 (`kmax_revisado` = 5 en `arboles_grandes_n22–24.json`) sin encontrar pares. Eso da Φ(5) ≥ 25 y resuelve la ambigüedad de SST.

**Otras truncaciones**
- **Heil–Ji** (ver A.1). La 2-truncación es el polinomio de matchings. En árboles equivale al característico, así que no basta (Remark 4.2 de Heil–Ji ✔).
- **Awan** (ver A.2): contar bosques con ≤ t aristas no basta si t < ⌊k/2⌋ ✔.

**Versión con pesos en vértices: Aliste-Prieto, Crew, Spirkl, Zamora (2021)** ✔. *A vertex-weighted Tutte symmetric function, and constructing graphs with equal chromatic symmetric function*, *Electron. J. Combin.* 28(2) (2021) P2.1 (DOI 10.37236/10018; arXiv:2007.11042). Leí el PDF.
1. Definen X_B con pesos en vértices. Prueban borrado–contracción y su equivalencia con el W-polinomio y el policromato.
2. Dan varias construcciones de pares con igual CSF o igual Tutte simétrica:
   - grafos «split»;
   - automorfismos;
   - relaciones modulares de Orellana–Scott;
   - una versión de Brylawski que da pares con cintura arbitrariamente grande.
3. **Sección 7: árboles con pesos.** Hay un método algebraico con grafos 2-punteados y composiciones de enteros (factorización irreducible única de Billera–Thomas–van Willigenburg). Con él construyen **familias arbitrariamente grandes de árboles con pesos con igual W-polinomio**, que ni siquiera son isomorfos sin pesos.
4. **Datos.** Revisaron todos los grafos de ≤ 8 vértices: en los ~1000 pares con igual CSF, todos los grafos tienen un triángulo.
- **Por qué sirve.** Muestra que con pesos sí hay colisiones entre árboles. El obstáculo para Stanley es exigir que todos los pesos valgan 1. Las técnicas de composiciones son las mismas que en las orugas (ver C.2).

**Otras referencias relacionadas**
- **Crew y Spirkl, borrado–contracción para la CSF con pesos.** *Eur. J. Combin.* 89 (2020) 103143 (arXiv:1910.11859). ? (solo vía las referencias de ACSZ y Semantic Scholar).
- **Orellana y Scott.** *Discrete Math.* 320 (2014) 1–14 (arXiv:1308.6005) ✔: relación modular, familia infinita de pares unicíclicos con igual CSF y clasificación de árboles de un centroide. La revista la tomé de las referencias de Heil–Ji.
- **Aliste-Prieto, Martin, Wagner, Zamora.** arXiv:2402.10333 ✔; según arXiv, «to appear» en Bull. LMS. Ver C.2.
- **Loebl–Sereni.** Ver A.2.

---

## B. Métodos de búsqueda de contraejemplos que mejoran o critican el Cross-Entropy de Wagner

### B.1 El método base y su reimplementación

**Wagner (2021)** ✔. A. Z. Wagner, *Constructions in combinatorics via neural networks*, arXiv:2104.14516.
- Usa el deep cross-entropy method para encontrar construcciones y contraejemplos: permanentes con patrones prohibidos (Brualdi–Cao) y autovalores de adyacencia y de distancia.
- **Para la tesis.** Todos sus ejemplos son de maximizar un invariante de un solo objeto. Ninguno busca un par con invariante igual, que es lo que exige Stanley.

**Ghebleh, Al-Yakoob, Kanso, Stevanović (2024), parte I** ✔. *Reinforcement learning for graph theory, I. Reimplementation of Wagner's approach*, arXiv:2403.18429. Leí el PDF.
- Lo reimplementan para que sea más legible, más estable y «mucho más rápido»: resultados en minutos en vez de horas o días.
- Dos observaciones clave:
  1. En CEM la recompensa **solo** sirve para elegir la élite; su valor no entra en el entrenamiento.
  2. Al converger, la mayoría de los grafos generados son isomorfos a un mismo óptimo local, y hay que inyectar aleatoriedad. Dicen además que «no hay garantía previa de éxito».
- **Por qué importa.** Que la población colapse a un solo objeto juega en contra de buscar pares, donde se necesita diversidad *(inferencia mía)*.
- **Parte II** (arXiv:2403.20055, cotas de Ramsey con CEM): ? (solo la vi en resultados de búsqueda).

### B.2 Búsqueda Monte Carlo y búsqueda clásica que le ganan a CEM

**Roucairol y Cazenave (2022)** ✔. *Refutation of Spectral Graph Theory Conjectures with Monte Carlo Search*, arXiv:2207.03343. Leí el PDF.
- Prueban NMCS, NRPA y Greedy-BFS sobre las conjeturas de Wagner.
- NRPA refuta la conjetura 1 en 0,1 s; con CEM tomaba «unas horas».
- Greedy-BFS refuta la conjetura 2 en 5 min, con una función de puntaje más informativa; CEM tomaba «unos días».
- Recomiendan generar árboles primero, porque el espacio es mucho menor.
- **Para la tesis.** NRPA es la alternativa directa a CEM: se adapta la política sin red y con reinicios. Conviene probarla como línea base. Para pares, la construcción secuencial tendría que producir dos árboles, o un árbol y su pareja *(inferencia mía)*.

**Roucairol y Cazenave (2024)** ✔ (abstract). arXiv:2409.18626.
- De 13 conjeturas ya refutadas de Graffiti, refutan 12 en segundos.
- Refutan también la 197, que estaba abierta.
- La versión ECAI 2025 no la abrí: ?
- **Para la tesis.** Confirma con más conjeturas que la búsqueda sin red le gana a CEM en tiempo. Igual que en 2022, todo es de un solo objeto.

**Vito y Stefanus (2023)** ✔ (abstract). arXiv:2306.07956.
- Proponen AMCS, una variante de MCTS. Dicen superar a los algoritmos existentes; contra cuáles exactamente: ?
- **Para la tesis.** Es otra búsqueda Monte Carlo sin aprendizaje profundo que sirve como línea base. Su aporte es menor que el de Roucairol–Cazenave.

**Mehrabian et al. (2024)** ✔. *Finding increasingly large extremal graphs with AlphaZero and tabu search*, arXiv:2311.03583, IJCAI 2024; 19 autores, incluido Wagner. Leí el PDF.
- «El cross-entropy method de Wagner rinde mucho peor que tabu search y que AlphaZero.»
- El tabu incremental le gana a CEM «por amplio margen» a medida que crece el grafo.
- AlphaZero empata con tabu. La mejora vino del currículo, no del aprendizaje: «introducing learning did not yield improvements».
- Preguntan si el objetivo de RL (retorno esperado) es el adecuado cuando basta encontrar **una** solución en un entorno determinista.
- **Por qué es la crítica más directa.** Es el mismo resultado de la tesis (RL ≈ red con lr = 0), publicado con Wagner como coautor.

**Parczyk, Pokutta, Spiegel, Szabó (2024)** ✔. *New Ramsey multiplicity bounds and search heuristics*, *Found. Comput. Math.* (2024), DOI 10.1007/s10208-024-09675-6 (arXiv:2206.04036). Leí la sección 5.3.
- RL (incluido CEM, con otras redes y otros métodos) rinde peor que recocido simulado (SA) y tabu (TS), en calidad y en tiempo:

| caso | TS | SA | CEM |
|---|---|---|---|
| grafo de 13 vértices | 1 s | 10 s | ~8 min |
| un grafo de Cayley | ~1 min | ~4 min | < 1 h |
| grafo de 27 vértices | rara vez | 30 s | no lo encontró |

- Según los autores, tabu «requiere esencialmente cero ajuste».
- Dicen que RL es «inherentemente derrochador»: la construcción paso a paso es artificial porque no hay adversario, y basta un solo camino de la raíz a una hoja.
- **Para la tesis.** Da cifras concretas de que tabu y recocido le ganan a CEM. Justifica usarlos como líneas base obligatorias en vez de, o además de, lr = 0.

### B.3 Aprendizaje encima de búsqueda local

**PatternBoost (2024)** ✔. F. Charton, J. S. Ellenberg, A. Z. Wagner, G. Williamson, *PatternBoost: Constructions in Mathematics with a Little Help from AI*, arXiv:2411.00566. Leí partes del PDF.
- Alterna dos fases: una local (búsqueda clásica, a menudo greedy) y una global (un transformer entrenado con las mejores soluciones, que siembra la siguiente fase local).
- Dicen que CEM, en su forma básica, escala mal: la red simple cuesta entrenarla con secuencias de más de unos cientos de tokens.
- La fase global sola resulta «desastrosamente mala».
- El rendimiento «varía según el problema».
- Encontraron un contraejemplo a una conjetura de 30 años: un subgrafo generador del 6-cubo con diámetro 6 y 81 aristas.
- **Lectura** *(inferencia mía)*. El aprendizaje agrega valor solo encima de una buena búsqueda local con puntaje graduado. Para colisiones, antes hay que diseñar ese puntaje.

### B.4 Búsqueda de programas con LLM

**FunSearch (2024)** ✔. B. Romera-Paredes et al., *Mathematical discoveries from program search with large language models*, *Nature* 625 (2024) 468–475, DOI 10.1038/s41586-023-06924-6 (verificado en PMC).
- Mejoró cap sets (un cap set de 512 en dimensión 8) y heurísticas de bin packing.
- Según el paper, funciona mejor con evaluador eficiente, **puntaje rico (no binario)** y un esqueleto de programa.
- **Por qué importa.** «Tienen igual U_5 sí o no» es casi binario. Habría que definir una distancia entre coeficientes *(inferencia mía)*.

**AlphaEvolve** ✔. A. Novikov et al., arXiv:2506.13131 (2025).
- Agente evolutivo que edita código con un evaluador automático. Entre sus resultados, la multiplicación 4×4 compleja con 48 multiplicaciones.
- **Para la tesis.** Lo más prometedor sería pedirle que evolucione un *programa que construya pares*, por ejemplo variantes de los productos de A_k/B_k, en vez de árboles sueltos. No conozco quien lo haya hecho para U_k *(inferencia mía)*. Requiere acceso a la herramienta o a un clon abierto.

**Georgiev, Gómez-Serrano, Tao, Wagner (2025)** ✔. *Mathematical exploration and discovery at scale*, arXiv:2511.02864.
- 67 problemas: AlphaEvolve igualó lo mejor conocido en la mayoría y mejoró en varios.
- **Para la tesis.** Es el estado del arte en escala. El abstract no reporta comparaciones con líneas base simples. No hay ningún problema de colisiones entre los que vi.

**Ellenberg, Fraser-Taliente, Harvey, Srivastava, Sutherland (2025)** ✔. *Generative modeling for mathematical discovery*, arXiv:2503.11061.
- Implementación abierta de funsearch para matemáticos, sin cómputo pesado.
- **Para la tesis.** Es la vía práctica para probar búsqueda de programas con LLM sin depender de DeepMind. El usuario modifica un trozo corto de Python.

**Gideoni, Risi, Gal (2026)** ✔ (abstract). *Simple baselines are competitive with code evolution*, arXiv:2602.16805.
- Dos líneas base simples igualan o superan a los pipelines de evolución de código en todos los dominios probados.
- En cotas matemáticas, «el espacio de búsqueda importa mucho más que el algoritmo». Las cifras «4/9» que vi en la búsqueda: ?
- **Para la tesis.** Es el argumento general detrás del resultado lr = 0: lo que decide es cómo se define el espacio de búsqueda (por ejemplo, restringir a bicentroidales o a productos de ramas), no el optimizador.

**RadonBench (2026)** ✔ (abstract). X. Xu et al., arXiv:2610.33259.
- Comparan extensión exhaustiva, búsqueda local greedy, PatternBoost, AlphaZero y AlphaEvolve en un problema abierto. Ninguno llega al objetivo.
- Greedy y AlphaEvolve quedan igual de cerca.
- **Para la tesis.** Es un modelo de cómo reportar resultados negativos: casi aciertos verificados bajo una misma regla de éxito, comparando métodos clásicos y con aprendizaje. Sirve para el capítulo de resultados.

**Algoritmos genéticos clásicos.** No encontré una línea base de algoritmo genético clásico (sin LLM) aplicada a este tipo de problemas. Lo más cercano: PatternBoost describe su fase global como «una forma de algoritmo genético con un transformer», y Ellenberg et al. describen funsearch como «algoritmo genético guiado por LLM». La comparación con líneas base clásicas que sí está publicada es contra tabu, recocido y greedy (Parczyk; Mehrabian; RadonBench).

### B.5 Otras referencias de esta sección

- **Angileri et al. (2024)** ?. *A Systematization of the Wagner Framework*, arXiv:2406.12667: cuatro juegos (Linear, Local, Global, Flip) en Gymnasium. Solo lo vi en resultados de búsqueda.
- **Línea base lr = 0.** No encontré un paper que haga exactamente esa ablación. La evidencia más cercana es Mehrabian et al. y Parczyk et al.

### B.6 Trabajos que citan a Wagner 2021 y a arXiv:2002.08186

- **Wagner 2021.** Semantic Scholar registra citationCount = 90 (consulta del 21-11-2026) y la lista devolvió esos 90. Google Scholar podría tener más; no lo revisé. Revisé los títulos de los 90 con palabras clave (tree, chromatic, isomorph, spectr, baseline, tabu…).
  - El único que junta ML con la CSF es G. Bérczi, J. Klüver, *Reinforcement Learning the Chromatic Symmetric Function*, arXiv:2410.19189 ✔. Usa RL para conjeturar una fórmula de los coeficientes de la CSF en **grafos de intervalo unitario**. No trata árboles ni Stanley.
  - **No encontré ningún trabajo que aplique RL o ML a la conjetura de árboles de Stanley.** Es una ausencia en esta búsqueda dirigida, no una prueba de que no exista.
- **arXiv:2002.08186.** Semantic Scholar lista 7 trabajos que lo citan, todos algebraicos (por ejemplo Loehr–Warrington, Aval–Djenabou–McNamara y Crew–Spirkl). Ninguno es de búsqueda computacional.
- **arXiv:1509.09210.** Lista 27; ninguno de ML.

---

## C. Métodos para buscar COLISIONES (dos objetos distintos con el mismo invariante)

### C.1 Enumeración exhaustiva con huella (hash) y refinamiento

**Heil y Ji (2019)** ✔. Ver A.1.
- Es una huella aleatoria del invariante: evaluar en puntos aleatorios módulo un primo, al estilo Schwartz–Zippel.
- Después vienen cubetas y refinamiento iterativo.
- Distinguir es exacto: dos evaluaciones distintas prueban que los invariantes son distintos. Lo probabilístico es solo la prueba de igualdad.
- **Plantilla para U_5** *(inferencia mía)*. Evaluar U_5 truncado en (x₁, …, x₆) aleatorios módulo q, usar el valor como clave de hash y verificar exactamente solo las colisiones.

**Smith, Smith, Tian (2015)** ✔. Ver A.3.
- Evaluación perezosa: ordenar por 1-cuts y calcular 2-cuts solo dentro de las clases empatadas.
- Generación con `geng` de nauty.

**Generación de árboles**
- R. A. Wright, B. Richmond, A. Odlyzko, B. D. McKay, *Constant time generation of free trees*, *SIAM J. Comput.* 15(2) (1986) 540–548 ✔ (Crossref). Genera todos los árboles libres sin repetición, en tiempo proporcional a su cantidad.
- nauty/Traces: B. McKay, A. Piperno, *J. Symbolic Comput.* 60 (2014) 94–112 ? (solo vía las referencias de SST).

**Tamaño del espacio** ✔ (OEIS A000055 y A000081)
- Árboles libres: 104.636.890 (n = 25); 5.469.566.585 (n = 29); 300.628.862.480 (n = 33); 823.779.631.721 (n = 34).
- n = 24: 39.299.897.
- La ventana 24–33 suma unos 4,74·10¹¹ árboles (cálculo mío con OEIS).
- Comparación con lo que recorrieron Heil–Ji: es unas 87 veces el número de árboles de n = 29, o unas 55 veces el total de árboles con ≤ 29 vértices (≈ 8,7·10⁹).
- Árboles con raíz: 634.847 (n = 17); 235.381 (n = 16); 87.811 (n = 15); 32.973 (n = 14); 12.486 (n = 13).

**Espectros: el precedente más grande de enumeración de colisiones**
- **Brouwer y Spence (2009)** ✔. *Cospectral graphs on 12 vertices*, *Electron. J. Combin.* 16 (2009) #N20 (DOI 10.37236/258).
  - Calcularon el polinomio característico de **todos** los grafos de 12 vértices: 165.091.172.592 grafos, 148.187.993.520 polinomios distintos y 31.067.572.481 grafos con pareja coespectral.
  - La fracción de grafos con pareja sube y empieza a bajar en n = 11.
  - Los dos autores computaron por separado y obtuvieron lo mismo.
  - La nota no describe el método, así que no le atribuyo hashing.
- **Haemers y Spence (2004).** *Enumeration of cospectral graphs*, *Eur. J. Combin.* 25(2) (2004) 199–211. ✔ la ficha (Crossref); el contenido, ? (solo vía fuentes secundarias):
  - según Brouwer–Spence, da los datos de n = 10 y 11;
  - según Abiad et al., da la cota inferior n³·g_{n−1}·(1/24 − o(1)) de grafos con pareja vía switching GM4.
- **Godsil y McKay (1976)**, *Some computational results on the spectra of graphs*, LNM 560: ? (solo vía las referencias de Brouwer–Spence).
- **Abiad, Van de Berg, Simoens (2025)** ✔. *Counting cospectral graphs obtained via switching*, arXiv:2503.08627.
  - Marco general para contar parejas producidas por switching, con enumeración hasta 10 vértices usando etiquetas canónicas.
  - GM4 en orden 10: unos 163,6 millones de grafos y ~1000 h de cómputo.
- **Por qué sirve** *(inferencia mía)*. Muestra la escala a la que la enumeración exhaustiva con invariantes funciona: unos 10¹¹ objetos. Es la misma escala que la ventana 24–33 para U_5.

### C.2 Construcciones algebraicas que generan pares (switchings e intercambios)

**Switching de Godsil–McKay**
- **Fuente.** C. D. Godsil, B. D. McKay, *Constructing cospectral graphs*, *Aequationes Math.* 25 (1982) 257–268, DOI 10.1007/BF02189621. Ficha ✔ (Crossref).
- **Definición** ✔ (vía Abiad et al.). Un conjunto X de 2k vértices, con G[X] regular y cada vértice de fuera con 0, k o 2k vecinos en X. Se invierten las adyacencias de los que tienen k.
- **Límite.** Wang–Yeats ✔ advierten que solo se aplica a grafos con una partición que cumpla esos requisitos.
- **En árboles** *(inferencia mía)*. G[X] regular y sin ciclos obliga a casi no tener aristas en X, y el switching suele crear ciclos. No es la herramienta natural para árboles.

**Intercambio de ramas («limbs») de Schwenk**
- **Fuente.** A. J. Schwenk, *Almost all trees are cospectral*, en *New Directions in the Theory of Graphs* (1973) 275–307. ? (no está en línea; la ficha y el contenido los vi en Wang–Yeats y en McKay 1977).
- **Contenido.** Hay un par de ramas con raíz de 9 vértices tal que reemplazar una por la otra no cambia el polinomio característico. Casi todos los árboles contienen esa rama, así que casi todos los árboles tienen pareja coespectral.

**McKay (1977)** ✔. B. D. McKay, *On the spectral characterisation of trees*, *Ars Combin.* 3 (1977) 219–232 (PDF del autor).
- **Teorema 4.2.** Hay dos árboles con raíz T₁ y T₂ tales que, para cualquier S, las coalescencias S·T₁ y S·T₂ son coespectrales en varias matrices a la vez, incluida la de distancias.
- Corolario: casi ningún árbol queda determinado por esos espectros.
- Una búsqueda por computador mostró que los únicos árboles de ≤ 18 vértices con matrices de distancia coespectrales son los del Teorema 4.2.

**Wang y Yeats** ✔. *Limbs and cospectral vertices in trees*, arXiv:1811.12344, *J. Combin.* 2026.
- Extienden Schwenk a árboles con pesos: casi todo árbol con pesos tiene pareja.

**El análogo para U_k: A_k/B_k** *(inferencia mía, construida sobre el Teorema 12 de arXiv:2002.08186)*.
- **No se conoce una versión exacta.** Un intercambio de ramas exacto que preservara U para todo S sería directamente un contraejemplo a Stanley, y no se conoce ninguno. Como el U-polinomio con raíz distingue árboles con raíz (Teorema 9), dos ramas distintas nunca tienen igual U^r. Aun así, eso no descarta por sí solo un intercambio que preserve U después de pegar.
- **Lo que sí se conoce.** La versión truncada: pares de ramas (A_k, B_k) cuya diferencia U^r(A) − U^r(B) solo tiene términos de **grado alto**. Al pegarlas mediante las fórmulas de producto, la diferencia queda fuera de U_m.
- **Programa de búsqueda que esto sugiere.**
  1. Enumerar pares de árboles con raíz pequeños (≤ 17 vértices: 634.847 árboles con raíz) y calcular el **grado mínimo de U^r(A) − U^r(B)**.
  2. Combinarlos con los Lemas 4 y 5.
- Eso es una biblioteca de «ramas casi intercambiables», como en Schwenk y McKay. A_k/B_k sería un caso particular.

**Inversión de factores de composiciones (orugas)** ✔.
- **Fuente.** J. Aliste-Prieto, J. L. Martin, J. D. Wagner, J. Zamora, *Chromatic symmetric functions and polynomial invariants of trees*, arXiv:2402.10333 (Bull. LMS 2024; el DOI 10.1112/blms.13144 lo saqué de Semantic Scholar y no lo abrí: ?).
- **Teorema 14.** Si se invierte uno de los factores irreducibles de la composición que codifica una oruga, se obtiene una oruga con el mismo polinomio de grados semigeneralizado, equivalente al polinomio de subárboles.
  - Con q factores no palíndromos, hay ≥ 2^{q−1} orugas no isomorfas equivalentes.
  - Los dos árboles más chicos son Cat(2,2,1,3,3) y Cat(2,3,2,1,3).
- Con esto prueban la conjetura de Eisenstat–Gordon (Corolario 16).
- Eisenstat y Gordon: *Non-isomorphic caterpillars with identical subtree data*, *Discrete Math.* 306(8–9) (2006) 827–830, DOI 10.1016/j.disc.2006.01.022. Ficha ✔ (Crossref); el contenido, vía AMWZ.
- **Por qué sirve.** Es la técnica que sí produce colisiones en un invariante de árboles más débil que la CSF. Para la CSF no puede funcionar en orugas, porque ahí está probada (Loebl–Sereni). Sí podría inspirar construcciones en otras familias.
- La misma idea de composiciones da las colisiones de árboles con pesos de ACSZ (A.3).

**Relaciones modulares** ✔.
- Orellana–Scott y Penaguiao. Según ACSZ, toda colisión de CSF entre grafos se conecta mediante la relación modular de triángulos.
- **En árboles** *(inferencia mía)*. Los árboles no tienen triángulos, así que esas construcciones locales no producen árboles directamente. Eso explica, en parte, por qué no hay ningún par conocido.

**Vía Prouhet–Tarry–Escott** ✔.
- **El vínculo.** Aliste-Prieto, de Mier y Zamora (2017) convierten soluciones PTE en árboles con igual U_k (A.3).
- **La búsqueda de soluciones PTE.** P. Borwein, P. Lisoněk, C. Percival, *Computational investigations of the Prouhet-Tarry-Escott problem*, *Math. Comp.* 72(244) (2003) 2063–2070 (PDF del autor):
  - buscan soluciones ideales simétricas hasta tamaño 12;
  - **reducen la dimensión de 2m+1 a m+1**: dada la mitad, calculan la otra mitad como raíces de un polinomio interpolado y verifican con Newton;
  - criban con los primos que deben dividir la constante C;
  - encontraron dos soluciones de tamaño 10 unas cien veces más chicas que la de Letac.
- **Por qué sirve** *(inferencia mía)*.
  - Es un ejemplo de colisión **construida resolviendo para la pareja** en vez de buscarla. El análogo sería, dada una mitad de árbol, despejar qué otra mitad iguala U_k.
  - En cambio, como vía para pares chicos de árboles parece peor que Y/Z: el paper de 2020 dice que esos árboles no son explícitos.

### C.3 Búsqueda genérica de colisiones (cumpleaños, encuentro a mitad de camino)

**van Oorschot y Wiener (1999)** ✔ (abstract en el PDF del autor; la ficha *J. Cryptology* 12(1) 1–28, DOI 10.1007/PL00003816, la vi en dblp y Springer vía búsqueda: ?). *Parallel collision search with cryptanalytic applications*.
- Paralelizan la búsqueda de colisiones en caminatas pseudoaleatorias con «puntos distinguidos». Lo aplican a logaritmo discreto, colisiones de hash y encuentro a mitad de camino.

**Mucha, Nederlof, Pawlewicz, Węgrzycki (2019)** ✔. *Equal-Subset-Sum faster than the meet-in-the-middle*, arXiv:1905.02424.
- Es el problema clásico de «dos objetos distintos con igual valor». El encuentro a mitad de camino da O*(3^{n/2}) y lo mejoran a O*(1,7088^n).

**Lectura para U_k** *(inferencia mía)*.
- **El cumpleaños aleatorio no sirve aquí.** Rinde cuando hay muchas colisiones, y en U_4 hay **un solo par** entre 5.623.756 árboles de 22 vértices (SST). Muestrear árboles al azar no le gana a enumerar todo.
- **Mismo motivo para el RL.** Cualquier búsqueda estocástica por debajo de los umbrales de A.3 es inútil por definición, y cerca de ellos el objetivo es una aguja en un pajar. Eso es consistente con que el RL no le gane a lr = 0.
- **Lo que sí sirve: enumeración restringida a árboles bicentroidales, armados desde una tabla de mitades.** Reutiliza los cálculos por mitades, como el encuentro a mitad de camino, pero es una enumeración y no un algoritmo de MITM en sentido estricto.
  1. Un árbol de 2m vértices con dos centroides es un par no ordenado de árboles con raíz de m vértices unidos por una arista. Esto es un hecho de enumeración estándar que no verifiqué en una fuente.
  2. Por los Lemas 2 y 5 de arXiv:2002.08186, U(R₁—R₂) = Φ(U^r(R₁)U^r(R₂)) + U(R₁)U(R₂). Es una función bilineal simétrica de los U^r truncados de las dos mitades.
  3. Y_{2,1} y Z_{2,1} son de este tipo: mitades de 17 vértices.
  4. Para n = 34 hay unos 2·10¹¹ pares de mitades (634.847²/2). Para n = 26 hay unos 7,8·10⁷ y para n = 30 unos 3,9·10⁹.
  5. Con U^r truncado precalculado para cada árbol con raíz, la búsqueda queda en hashear U_5 de cada par. Es factible al menos hasta n ≈ 30, y en n = 34 con disco o cómputo distribuido.
- **Tamaños impares.** SST observó que su par impar de 4-cuts (n = 23) viene de subdividir la arista central del par de 22. Conjeturan que eso pasa en general. Subdividir sugiere cómo atacar los n impares, pero no los cubre: puede haber pares impares de otro tipo, y para descartarlos hace falta la enumeración completa (vía 1).

---

## Las 5 vías más prometedoras según la literatura

1. **Enumeración exhaustiva con huella y refinamiento, restringida a U_5 en la ventana 24 ≤ n ≤ 33.** Al estilo Heil–Ji y SST, con U^r truncado por programación dinámica sobre árboles con raíz.
   - Decide la Conjetura 15 para m = 5, que es la primera instancia no verificada.
   - Si aparece un par de menos de 34 vértices, la conjetura cae. Si no aparece, 34 queda como mínimo.
   - Tamaño (estimación mía con OEIS): unos 4,74·10¹¹ árboles. Son ~87 veces los árboles de n = 29, o ~55 veces todos los árboles con ≤ 29 vértices que recorrieron Heil–Ji.
2. **Enumeración restringida a árboles bicentroidales con tabla de mitades** (C.3).
   - Es el atajo para la vía 1: n = 26 son ~7,8·10⁷ pares de mitades y n = 30 son ~3,9·10⁹.
   - Se apoya en la observación de SST (los mínimos son intercambios de ramas entre centroides) y en que Y/Z tienen esa forma.
   - Encontraría todo par bicentroidal, pero no prueba minimalidad sobre todos los árboles.
3. **Biblioteca de «ramas casi intercambiables» al estilo Schwenk y McKay** (C.2).
   - Buscar pares de árboles con raíz pequeños cuya diferencia U^r(A) − U^r(B) tenga grado mínimo alto, y combinarlos con los Lemas 4 y 5.
   - Generaliza A_k/B_k y puede dar pares con igual U_m más chicos que Y/Z, o construcciones nuevas.
4. **Si se quiere aprendizaje, ponerlo encima de una búsqueda local fuerte, no en lugar de ella.**
   - Primero tabu o recocido como línea base (Parczyk et al.; Mehrabian et al.), luego NRPA (Roucairol–Cazenave) y PatternBoost.
   - Medirlo contra las familias conocidas de SST: por ejemplo, encontrar las 11 familias de 3-cuts en n = 18 o las 92 en n = 20.
   - Usar un puntaje graduado (distancia entre vectores de coeficientes de U_k) y no binario.
   - La literatura (Mehrabian; Parczyk; Gideoni et al.) anticipa que el espacio de búsqueda pesará más que el algoritmo.
5. **Podar con lo ya probado y con invariantes baratos.**
   - Excluir orugas, arañas, diámetro ≤ 5, árboles con dos vértices de grado ≥ 3 y las clases de Zeng y Awan.
   - Usar como claves previas los invariantes que la CSF determina: sucesiones de grados y de caminos (MMW), tamaño del núcleo y caminos colgantes (Crew), sucesión de grados generalizada (AMWZ).
   - Aprovechar la reducción a árboles k-pesados de Orellana–Tomba para ignorar los coeficientes con partes 1.

---

## Método y límites de esta revisión

- **Cómo verifiqué.** Abrí las fuentes en arXiv (abs, HTML o PDF pasado a texto con `pdftotext`), en Crossref (fichas por DOI), en PMC, en los PDF de los autores y en OEIS. Los papers clave los leí enteros o en las secciones citadas: 2002.08186, Heil–Ji, SST, ACSZ, AMWZ, Mehrabian, Parczyk, Roucairol–Cazenave, PatternBoost, Ghebleh, McKay 1977, Brouwer–Spence y Borwein et al.
- **Cobertura.** La búsqueda fue dirigida, no sistemática. Para las citas usé Semantic Scholar: 90 de Wagner 2021, 7 de 2002.08186 y 27 de 1509.09210.
- **Quién la hizo.** Un agente de investigación con búsqueda web. Las marcas ✔ son suyas; las afirmaciones centrales para el plan (Mehrabian et al., Parczyk et al., Heil–Ji y arXiv:2002.08186) conviene releerlas antes de citarlas en el escrito.
