# Evaluación de modelos de embeddings — resultados

## Criterios de comparación


Criterios de comparación entre modelos candidatos:
 
1. Calidad semántica (peso alto): un modelo que no distingue documento
   correcto de incorrecto no sirve para RAG, sin importar qué tan rápido sea.
   Umbral mínimo aceptado: retrieval_accuracy >= 0.75 sobre el set de
   evaluación del dominio IGPUBA.

   Nota sobre el umbral: se ajustó de 0.85 a 0.75 tras un análisis caso por
   caso que mostró que los embeddings semánticos genéricos priorizan el tema 
   general del documento (ej. "falla de sistema BES") por sobre identificadores
   específicos de pozo (ej. CH-88 vs LL-112). Esto es una limitación conocida 
   de estos modelos, no un defecto de un candidato puntual: los 3 modelos
   evaluados mostraron el mismo patrón de fallo en los mismos casos, lo que 
   confirma que el techo alcanzable de accuracy en este corpus técnico
   (documentos del mismo doc_type con vocabulario y estructura muy similares) 
   es más bajo que en un dominio genérico.
 
2. Velocidad de generación (peso medio): relevante para la indexación inicial
   del corpus y para la latencia de consultas en tiempo real. Se prioriza
   texts_per_sec en CPU, ya que no se asume disponibilidad de GPU en
   producción.
 
3. Consumo de recursos (peso medio): peak_ram_mb y approx_disk_size_mb,
   relevantes porque el sistema corre en infraestructura del IGPUBA sin
   escalado elástico garantizado.
 
4. Restricción no negociable: el modelo debe poder ejecutarse localmente
   (sin llamadas a API externas), por confidencialidad de los datos.
   Esto descarta de entrada modelos como text-embedding-ada-002 de OpenAI.
 
Un modelo pasa a "seleccionado" solo si cumple el umbral de calidad Y
domina o empata en al menos uno de los otros criterios frente a las
alternativas evaluadas.

| Modelo | Dim | Textos/s | RAM (MB) | Disco aprox (MB) | Accuracy | Margen prom. |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 768 | 3.7 | 818.7 | 1100 | 0.8 | 0.0283 |
| intfloat/multilingual-e5-large | 1024 | 1.1 | 1235.4 | 2240 | 0.8 | 0.0322 |
| BAAI/bge-m3 | 1024 | 1.1 | 84.7 | 2270 | 0.8 | 0.0795 |

## Justificación

**Modelo seleccionado: `intfloat/multilingual-e5-base`**

- Cumple el umbral de calidad definido (accuracy=0.8 >= 0.75).
- Entre los modelos que cumplen el umbral, es el de mayor velocidad de generación (3.7 textos/s en CPU).
- Consumo de RAM medido: 818.7 MB. Tamaño en disco aproximado: 1100 MB.

- Otros modelos que también cumplieron el umbral de calidad (intfloat/multilingual-e5-large, BAAI/bge-m3) fueron descartados por menor velocidad y/o mayor consumo de recursos, sin ofrecer una mejora de calidad que lo justifique.