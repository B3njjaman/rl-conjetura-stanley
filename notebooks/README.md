# Notebooks

Cuadernos de trabajo de la tesis. Cada cambio respecto al codigo de Wagner esta
marcado en el codigo como `# >>> CAMBIO N vs Wagner: ...` y documentado en
[`../CAMBIOS_VS_WAGNER.md`](../CAMBIOS_VS_WAGNER.md).

| Notebook | Descripcion | Cambios vs Wagner |
|---|---|---|
| `01_validacion_wagner_conj21.ipynb` | Validacion del metodo (Cross-Entropy + red) reproduciendo la conjetura 2.1 de Wagner: lambda_1 + mu >= sqrt(n-1) + 1. | 1-9 |
| `02_pares_de_grafos.ipynb` | Duplica el algoritmo para generar *pares* de grafos no isomorfos (una palabra de largo 2*C(N,2)). Primer paso hacia comparar el U0-polinomio de Stanley. | 10-12 |

> Los checkpoints y sesiones (`datos/`, `*.pkl`, `*.pt`) no se versionan (ver `../.gitignore`).
