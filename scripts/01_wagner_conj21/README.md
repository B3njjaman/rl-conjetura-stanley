# Scripts del notebook 01 (validacion con la conjetura 2.1 de Wagner)

El notebook `notebooks/01_validacion_wagner_conj21.ipynb` es la fuente de verdad del
algoritmo. Estos scripts existen para correrlo como corridas largas, varias a la vez,
y para dejar los resultados ordenados en `resultados/01_wagner_conj21/`.

| Script | Que hace |
|---|---|
| `exportar_notebook.py` | Saca las celdas de codigo del notebook a `nb01_core.py` (generado, no editar a mano). Correrlo cada vez que cambie el notebook. |
| `nb01_core.py` | El notebook como modulo: red, matching, recompensa, generacion de sesiones, `entrenar`. |
| `variantes.py` | El ciclo de entrenamiento con dos modos: `notebook` (tal cual el notebook) y `wagner_fit` (SGD 1e-4 por minilotes de 32 + seleccion con tope, como el `model.fit` de Wagner). Registra elite/super y cuantos grafos distintos hay. |
| `correr.py` | Corre una instancia y deja `log.txt`, `historial.csv`, `linea_tiempo.json`, `mejor_grafo.json`, `resumen.json` en `resultados/01_wagner_conj21/<modo>_seed<N>/`. Los `.pkl`/`.pt` quedan ahi mismo pero no se versionan. |
| `enumerar_arboles.py` | Verdad de terreno: evalua el score de los 317 955 arboles de 19 vertices (el minimo de lambda1+mu sobre conexos se alcanza en un arbol). |
| `graficar.py` | Figuras para el reporte: `recompensa.png`, `mejor_grafo.png`, `linea_tiempo.png` por corrida; `comparacion_corridas.png`, `resumen_corridas.md` e histograma de la verdad de terreno. |

## Como correr

Desde la raiz del repo:

```bash
python scripts/01_wagner_conj21/exportar_notebook.py
python scripts/01_wagner_conj21/correr.py --modo notebook   --seed 0 --iters 1000 --hilos 2
python scripts/01_wagner_conj21/correr.py --modo wagner_fit --seed 0 --iters 3000 --tiempo-max 3600 --hilos 2
python scripts/01_wagner_conj21/enumerar_arboles.py
python scripts/01_wagner_conj21/graficar.py
```

Una iteracion (1000 sesiones, N=19) tarda ~1.2 s en CPU con 4 hilos; con `--hilos 2`
se pueden lanzar 4 corridas en paralelo en una maquina de 8 nucleos.
