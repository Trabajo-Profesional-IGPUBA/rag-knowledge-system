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

""" 
Modelos como all-MiniLM-L6-v2, se consideraron pero al ser solo inglés, quedo descartado.
"""

CANDIDATE_MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "notes": "Candidato actual: liviano, multilingüe.",
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "notes": "Fuerte en retrieval multilingüe según benchmarks MTEB.",
    },
    {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "notes": "Mayor calidad esperada, ~2x más pesado que el actual.",
    },
]
 
# Tamaño en disco, ya que medirlo en runtime requiere acceso al cache dir y no es 100% portable.
APPROX_DISK_SIZE_MB = {
    "paraphrase-multilingual-MiniLM-L12-v2": 470,
    "intfloat/multilingual-e5-base": 1100,
    "paraphrase-multilingual-mpnet-base-v2": 970,
}
 
# Dataset query -> doc correcto / doc incorrecto, usado para medir calidad semántica
# TODO: reemplazar/ampliar con 30-50 casos reales del dominio IGPUBA antes
# de tomar la decisión final; con pocos casos el accuracy no es representativo.
EVALUATION_QUERIES: list[dict] = [
    {
        "query": "¿Cómo se tramita una licencia de conducir?",
        "doc_correcto": "Requisitos y pasos para obtener o renovar la licencia de conducir.",
        "doc_incorrecto": "Trámite de baja de vehículo ante el registro correspondiente.",
    },
    {
        "query": "Requisitos para inscribirse en el padrón electoral",
        "doc_correcto": "Documentación necesaria para el registro en el padrón de votantes.",
        "doc_incorrecto": "Requisitos para participar en el consejo consultivo vecinal.",
    },
    {
        "query": "¿Dónde reclamo por un bache en la calle?",
        "doc_correcto": "Canal de reclamos por infraestructura vial y baches.",
        "doc_incorrecto": "Canal para reportar fallas en el alumbrado público de la vía pública.",
    },
    {
        "query": "Certificado de discapacidad, cómo solicitarlo",
        "doc_correcto": "Trámite y requisitos para obtener el certificado único de discapacidad.",
        "doc_incorrecto": "Requisitos para solicitar el certificado de buena conducta.",
    },
    {
        "query": "Pago de tasas municipales online",
        "doc_correcto": "Plataforma de pago digital de tasas y contribuciones municipales.",
        "doc_incorrecto": "Requisitos para solicitar la exención de tasas municipales para jubilados.",
    },
    {
        "query": "¿Cómo pido turno para el registro civil?",
        "doc_correcto": "Sistema de turnos online para trámites en el registro civil.",
        "doc_incorrecto": "Documentación y turno necesarios para contraer matrimonio civil.",
    },
    {
        "query": "Habilitación comercial para abrir un local",
        "doc_correcto": "Pasos y documentación para tramitar la habilitación comercial de un local.",
        "doc_incorrecto": "Requisitos para tramitar el traslado o mudanza de un comercio habilitado.",
    },
    {
        "query": "Denunciar ruidos molestos de un vecino",
        "doc_correcto": "Procedimiento para realizar una denuncia por ruidos molestos.",
        "doc_incorrecto": "Procedimiento para realizar una denuncia por maltrato animal.",
    },
    {
        "query": "Exención de tasas para jubilados",
        "doc_correcto": "Requisitos para solicitar la exención de tasas municipales para jubilados.",
        "doc_incorrecto": "Requisitos y trámite para acceder a la pensión no contributiva municipal.",
    },
    {
        "query": "¿Cómo doy de baja un vehículo?",
        "doc_correcto": "Trámite de baja de vehículo ante el registro correspondiente.",
        "doc_incorrecto": "Trámite de transferencia de titularidad de un vehículo usado.",
    },
    {
        "query": "Inscripción a jardines maternales municipales",
        "doc_correcto": "Requisitos e inscripción online a jardines maternales municipales.",
        "doc_incorrecto": "Requisitos e inscripción a la escuela primaria municipal para el ciclo lectivo.",
    },
    {
        "query": "¿Qué documentación necesito para casarme por civil?",
        "doc_correcto": "Documentación y turno necesarios para contraer matrimonio civil.",
        "doc_incorrecto": "Documentación necesaria para tramitar el divorcio de común acuerdo.",
    },
    {
        "query": "Reclamo por falta de alumbrado público",
        "doc_correcto": "Canal para reportar fallas en el alumbrado público de la vía pública.",
        "doc_incorrecto": "Canal de reclamos por infraestructura vial y baches.",
    },
    {
        "query": "¿Cómo solicito una audiencia con un funcionario?",
        "doc_correcto": "Procedimiento para solicitar audiencia con autoridades municipales.",
        "doc_incorrecto": "Procedimiento para presentar un reclamo ante la defensoría del pueblo.",
    },
    {
        "query": "Permiso para realizar un evento en la vía pública",
        "doc_correcto": "Requisitos para tramitar permiso de uso del espacio público para eventos.",
        "doc_incorrecto": "Requisitos para tramitar permiso de obra en construcción sobre vía pública.",
    },
    {
        "query": "Certificado de residencia, cómo lo obtengo",
        "doc_correcto": "Trámite para obtener el certificado de residencia municipal.",
        "doc_incorrecto": "Trámite para actualizar el domicilio registrado en el padrón municipal.",
    },
    {
        "query": "Subsidio municipal por desempleo",
        "doc_correcto": "Requisitos para acceder al subsidio municipal por desempleo.",
        "doc_incorrecto": "Requisitos para acceder a becas municipales para estudiantes universitarios.",
    },
    {
        "query": "Cómo tramitar el boleto estudiantil",
        "doc_correcto": "Requisitos e inscripción para obtener el boleto de transporte estudiantil.",
        "doc_incorrecto": "Requisitos para obtener la tarjeta de transporte gratuito para jubilados.",
    },
    {
        "query": "Solicitar poda de un árbol en la vereda",
        "doc_correcto": "Procedimiento para solicitar la poda de árboles en el espacio público.",
        "doc_incorrecto": "Procedimiento para solicitar autorización de tala de un árbol en propiedad privada.",
    },
    {
        "query": "Renovación del DNI, dónde se hace",
        "doc_correcto": "Trámite y turno para la renovación del DNI en oficinas del registro civil.",
        "doc_incorrecto": "Trámite y turno para la tramitación del pasaporte en oficinas del registro civil.",
    },
    {
        "query": "¿Cómo inicio el trámite de jubilación municipal?",
        "doc_correcto": "Requisitos y documentación para iniciar el trámite de jubilación municipal.",
        "doc_incorrecto": "Requisitos y documentación para solicitar una pensión por invalidez municipal.",
    },
    {
        "query": "Inscripción a ferias itinerantes de emprendedores",
        "doc_correcto": "Requisitos para inscribirse como expositor en ferias itinerantes municipales.",
        "doc_incorrecto": "Requisitos para inscribirse en talleres municipales de oficios y capacitación.",
    },
    {
        "query": "Horarios y trámites en la biblioteca pública municipal",
        "doc_correcto": "Horarios de atención y servicios de la biblioteca pública municipal.",
        "doc_incorrecto": "Horarios de atención y servicios del centro cultural municipal.",
    },
    {
        "query": "Cómo pedir un traslado en ambulancia municipal",
        "doc_correcto": "Procedimiento para solicitar traslados en ambulancia del sistema de salud municipal.",
        "doc_incorrecto": "Procedimiento para solicitar turno en el centro de salud municipal más cercano.",
    },
    {
        "query": "Dónde consultar el boletín oficial municipal",
        "doc_correcto": "Acceso y consulta del boletín oficial municipal con ordenanzas vigentes.",
        "doc_incorrecto": "Acceso y consulta del registro de proveedores habilitados del municipio.",
    },
    {
        "query": "Cómo hacer un reclamo como consumidor",
        "doc_correcto": "Procedimiento para presentar un reclamo en la oficina de defensa del consumidor.",
        "doc_incorrecto": "Procedimiento para presentar una denuncia en la oficina de defensa del vecino.",
    },
    {
        "query": "Permiso de obra para construcción de vivienda",
        "doc_correcto": "Requisitos para tramitar el permiso de obra de construcción de vivienda unifamiliar.",
        "doc_incorrecto": "Requisitos para tramitar el permiso de demolición de una construcción existente.",
    },
    {
        "query": "Cómo tramitar la libreta sanitaria",
        "doc_correcto": "Requisitos y turno para tramitar la libreta sanitaria municipal.",
        "doc_incorrecto": "Requisitos y turno para tramitar el carnet de manipulador de alimentos.",
    },
    {
        "query": "Solicitar el retiro de residuos voluminosos",
        "doc_correcto": "Procedimiento para solicitar el retiro municipal de residuos voluminosos.",
        "doc_incorrecto": "Cronograma habitual de recolección de residuos domiciliarios por barrio.",
    },
    {
        "query": "Cómo inscribirme para votar por primera vez",
        "doc_correcto": "Requisitos para el primer empadronamiento de votantes que alcanzan la mayoría de edad.",
        "doc_incorrecto": "Documentación necesaria para el registro en el padrón de votantes ya inscriptos.",
    },
]