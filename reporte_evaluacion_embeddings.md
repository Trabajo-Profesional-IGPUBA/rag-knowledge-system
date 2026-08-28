# Evaluación de modelos de embeddings — resultados

| Modelo | Dim | Textos/s | RAM (MB) | Disco aprox (MB) | Accuracy | Margen prom. |
|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 48.6 | 756.9 | 470 | 0.8 | 0.1052 |
| intfloat/multilingual-e5-base | 768 | 6.1 | 344.7 | 1100 | 0.8 | 0.0277 |
| paraphrase-multilingual-mpnet-base-v2 | 768 | 16.4 | -9.9 | 970 | 0.8 | 0.0979 |

## Justificación

**Modelo seleccionado: `paraphrase-multilingual-MiniLM-L12-v2`**

- Cumple el umbral de calidad definido (accuracy=0.8 >= 0.75).
- Entre los modelos que cumplen el umbral, es el de mayor velocidad de generación (48.6 textos/s en CPU).
- Consumo de RAM medido: 756.9 MB. Tamaño en disco aproximado: 470 MB.

- Otros modelos que también cumplieron el umbral de calidad (intfloat/multilingual-e5-base, paraphrase-multilingual-mpnet-base-v2) fueron descartados por menor velocidad y/o mayor consumo de recursos, sin ofrecer una mejora de calidad que lo justifique.