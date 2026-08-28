# Evaluación de modelos de embeddings — resultados

| Modelo | Dim | Textos/s | RAM (MB) | Disco aprox (MB) | Accuracy | Margen prom. |
|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 273.3 | 733.6 | 470 | 1.0 | 0.2497 |
| intfloat/multilingual-e5-base | 768 | 89.4 | 319.2 | 1100 | 1.0 | 0.0551 |
| paraphrase-multilingual-mpnet-base-v2 | 768 | 88.0 | 44.7 | 970 | 0.967 | 0.2151 |

## Justificación

**Modelo seleccionado: `paraphrase-multilingual-MiniLM-L12-v2`**

- Cumple el umbral de calidad definido (accuracy=1.0 >= 0.85).
- Entre los modelos que cumplen el umbral, es el de mayor velocidad de generación (273.3 textos/s en CPU).
- Consumo de RAM medido: 733.6 MB. Tamaño en disco aproximado: 470 MB.

- Otros modelos que también cumplieron el umbral de calidad (intfloat/multilingual-e5-base, paraphrase-multilingual-mpnet-base-v2) fueron descartados por menor velocidad y/o mayor consumo de recursos, sin ofrecer una mejora de calidad que lo justifique.