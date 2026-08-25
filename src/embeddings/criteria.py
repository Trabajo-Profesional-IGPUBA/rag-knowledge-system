"""
Criterios de evaluación para la comparación de modelos de embeddings.
 
Historia: "Definición de criterios de evaluación"
  - QUALITY_METRICS: métricas de calidad identificadas.
  - PERFORMANCE_METRICS: métricas de rendimiento identificadas.
  - RESOURCE_METRICS: métricas de consumo de recursos identificadas.
  - COMPARISON_CRITERIA: documentación de los criterios de comparación.
"""
 
QUALITY_METRICS = {
    "retrieval_accuracy": (
        "Proporción de consultas de evaluación donde el modelo asigna mayor "
        "similitud coseno al documento correcto que al incorrecto (top-1)."
    ),
    "avg_margin": (
        "Diferencia promedio entre la similitud del documento correcto y la "
        "del incorrecto. Mide qué tan 'segura' es la separación, no solo si "
        "acierta."
    ),
    "avg_similarity_correct": (
        "Similitud coseno promedio entre consulta y documento correcto. "
        "Sirve para detectar modelos que compriman todo el espacio vectorial "
        "(similitudes altas para todo, incluso lo irrelevante)."
    ),
}
 
PERFORMANCE_METRICS = {
    "load_time_sec": "Tiempo de carga del modelo en memoria.",
    "encode_time_sec": "Tiempo total para generar embeddings del set de prueba.",
    "texts_per_sec": "Throughput: cantidad de textos vectorizados por segundo.",
    "embedding_dim": "Dimensión del vector resultante (afecta storage e índice).",
}
 
RESOURCE_METRICS = {
    "peak_ram_mb": (
        "Incremento de memoria residente (RSS) del proceso entre antes y "
        "después de cargar el modelo + generar embeddings."
    ),
    "approx_disk_size_mb": (
        "Tamaño aproximado del modelo en disco (documentado manualmente por "
        "modelo, según la ficha publicada en Hugging Face)."
    ),
}
 
COMPARISON_CRITERIA = """
Criterios de comparación entre modelos candidatos:
 
1. Calidad semántica (peso alto): un modelo que no distingue documento
   correcto de incorrecto no sirve para RAG, sin importar qué tan rápido sea.
   Umbral mínimo aceptado: retrieval_accuracy >= 0.85 sobre el set de
   evaluación del dominio IGPUBA.
 
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
"""


CANDIDATE_MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "notes": "Candidato actual: liviano, multilingüe.",
    },
    {
        "name": "all-MiniLM-L6-v2",
        "notes": "Solo inglés — se espera bajo desempeño en consultas en español.",
    },
    {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "notes": "Mayor calidad esperada, ~2x más pesado que el actual.",
    },
]
 
# Tamaño en disco, ya que medirlo en runtime requiere acceso al cache dir y no es 100% portable.
APPROX_DISK_SIZE_MB = {
    "paraphrase-multilingual-MiniLM-L12-v2": 470,
    "all-MiniLM-L6-v2": 90,
    "paraphrase-multilingual-mpnet-base-v2": 970,
}
 
# Dataset query -> doc correcto / doc incorrecto, usado para medir calidad semántica
# TODO: reemplazar/ampliar con 30-50 casos reales del dominio IGPUBA antes
# de tomar la decisión final; con pocos casos el accuracy no es representativo.
EVALUATION_QUERIES: list[dict] = [
    {
        "query": "¿Cómo se tramita una licencia de conducir?",
        "doc_correcto": "Requisitos y pasos para obtener o renovar la licencia de conducir.",
        "doc_incorrecto": "Cronograma de recolección de residuos por barrio.",
    },
    {
        "query": "Requisitos para inscribirse en el padrón electoral",
        "doc_correcto": "Documentación necesaria para el registro en el padrón de votantes.",
        "doc_incorrecto": "Horarios de atención de las oficinas de turismo.",
    },
    {
        "query": "¿Dónde reclamo por un bache en la calle?",
        "doc_correcto": "Canal de reclamos por infraestructura vial y baches.",
        "doc_incorrecto": "Formulario de inscripción a talleres culturales municipales.",
    },
    {
        "query": "Certificado de discapacidad, cómo solicitarlo",
        "doc_correcto": "Trámite y requisitos para obtener el certificado único de discapacidad.",
        "doc_incorrecto": "Listado de ferias de emprendedores del mes.",
    },
    {
        "query": "Pago de tasas municipales online",
        "doc_correcto": "Plataforma de pago digital de tasas y contribuciones municipales.",
        "doc_incorrecto": "Reglamento interno de uso de espacios verdes.",
    },
]
 