# Resultados: validacion con la conjetura 2.1 de Wagner (N = 19)

Todo lo de esta carpeta sale de `scripts/01_wagner_conj21/` (ver su README para
los comandos). Los archivos pesados (`.pkl`, `.pt`, `.npy`) quedan en local y no se
versionan; lo que esta en el repo (CSV, JSON, PNG, MD) alcanza para rehacer las
figuras y las tablas del reporte.

## Que buscamos

Refutar lambda_1(G) + mu(G) >= sqrt(n-1) + 1 con n = 19, como Wagner (2021). El score
es sqrt(18) + 1 - lambda_1 - mu; un score > 0 es un contraejemplo. El contraejemplo de
Wagner (dos estrellas K_{1,8} unidas por un vertice intermedio) da +0.0804.

## Subcarpetas

| Carpeta | Contenido |
|---|---|
| `verdad_terreno_arboles19/` | Score de **todos** los arboles no isomorfos de 19 vertices (317 955). `resumen.json`, `top_arboles.json` (los 11 con score > -0.5), `histograma.json`, `histograma_scores.png`, `contraejemplo_wagner.png`. |
| `notebook_seed0/`, `notebook_seed1/` | El algoritmo **tal cual esta en el notebook 01** (Adam 1e-2, un paso full-batch por iteracion), semillas 0 y 1. |
| `wagner_fit_seed0/`, `wagner_fit_seed1/` | Variante con el entrenamiento **como el `model.fit` de Wagner** (SGD 1e-4, minilotes de 32, 1 epoca) y su seleccion con tope. |

Cada carpeta de corrida tiene: `log.txt` (una linea por iteracion), `historial.csv`
(mejor, top-100, loss, tamano de elite/super, grafos distintos), `linea_tiempo.json`
(mejor palabra cada 20 iteraciones), `mejor_grafo.json` (decodificado y verificado con
NetworkX), `resumen.json`, y las figuras `recompensa.png`, `mejor_grafo.png`,
`linea_tiempo.png`.

## Figuras globales

- `comparacion_corridas.png`: mejor score acumulado de todas las corridas y cuantos
  grafos distintos quedan entre las super-sesiones (muestra el colapso de la politica).
- `resumen_corridas.md`: tabla con el mejor score, la iteracion y el grafo de cada corrida.

## Verdad de terreno (resultado)

| Dato | Valor |
|---|---|
| Arboles de 19 vertices | 317 955 |
| Score maximo | +0.0804 (el grafo de Wagner: lambda_1 = sqrt(10), mu = 2) |
| Arboles con score > 0 | 2 |
| Arboles con score > -0.5 | 11 |
| Estrella K_{1,18} | score exactamente 0 (es el caso de igualdad de la conjetura) |
| Mediana del score | -4.37 |

Como el minimo de lambda_1 + mu sobre grafos conexos se alcanza en un arbol, +0.0804 es
el maximo global del score para n = 19: ninguna corrida puede superarlo, y encontrar el
contraejemplo significa dar con 1 de 2 arboles entre 317 955 (y entre 2^171 palabras).
