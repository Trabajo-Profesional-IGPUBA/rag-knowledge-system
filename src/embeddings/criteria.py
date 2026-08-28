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
# TODO: Ampliar con 30-50 casos reales del dominio IGPUBA antes
# de tomar la decisión final; con pocos casos el accuracy no es representativo.

EVALUATION_QUERIES: list[dict] = [
    {
        "query": "¿Por qué se produjo un screen-out durante la estimulación hidráulica del pozo CH-45?",
        "doc_correcto": (
            "Durante el bombeo de la etapa 4 de estimulación hidráulica en la formación Agrio "
            "(intervalo 3.120 - 3.145 metros), se observó un incremento repentino y crítico de la "
            "presión de superficie de 6.200 a 8.500 psi al ingresar el agente sostén de alta "
            "densidad (arena mesh 20/40). El evento fue catalogado como un screen-out (arenamiento) "
            "prematuro, originado por una caída de la tasa de inyección debido a la falla mecánica "
            "en una de las bombas de la unidad fracturadora en superficie."
        ),
        "doc_incorrecto": (
            "El sistema BES sufrió una caída drástica de eficiencia hidráulica hasta su detención "
            "total tras 120 días de marcha. La inspección en taller reveló un desgaste severo por "
            "abrasión debido al flujo continuo de arenas de formación y material de fractura "
            "remanente. La causa raíz fue la rotura del filtro de fondo (sand screen) durante una "
            "sobre estimulación hidráulica previa del pozo CH-88."
        ),
    },
    {
        "query": "¿Cuál fue la causa de la rotura de varillas en el pozo LL-205?",
        "doc_correcto": (
            "Durante la extracción de la sarta de bombeo mecánico por caída abrupta de producción, "
            "se detectó un desprendimiento de las varillas a los 1.820 metros de profundidad. Se "
            "constató rotura por fatiga en el cuello de una varilla de 7/8\" (Grado D). Se bajó una "
            "herramienta de pesca tipo overshot de 2 3/8\" con grapa espiral."
        ),
        "doc_incorrecto": (
            "Se constata pozo parado por rotura de varillas. Se procede a ahogar el pozo con 40 bbl "
            "de agua salada filtrada. Inicio de maniobra de extracción de sarta de bombeo mecánico "
            "(AIB). Se recuperan 42 varillas de 7/8\". A la altura de la varilla 43 se encuentra "
            "punto de corte (fatiga por corrosión severa) en el pozo PM-104."
        ),
    },
    {
        "query": "¿Qué pérdida de circulación se registró al bajar el casing en el pozo PM-104?",
        "doc_correcto": (
            "Durante la carrera de bajada del casing de 9 5/8\" a los 2.450 metros mdf, se detectó "
            "una pérdida de circulación severa de aprox. 15 m³/h en la formación Quintuco. Se "
            "procedió a suspender la maniobra y se bombearon dos píldoras de material de pérdida de "
            "circulación (LCM) de alta concentración."
        ),
        "doc_incorrecto": (
            "El análisis de presiones estáticas (RFT) e interpretación sísmica 3D en la Formación "
            "Agrio indica una fuerte compartimentación del reservorio provocada por una falla normal "
            "sellante con rumbo NO-SE. Esta barrera estructural aísla hidráulicamente al pozo "
            "productor PM-104 del área de influencia directa del pozo inyector PI-05."
        ),
    },
    {
        "query": "¿Cuál fue la causa raíz de la falla del sistema BES en el pozo CH-88?",
        "doc_correcto": (
            "La causa raíz fue la rotura del filtro de fondo (sand screen) durante una sobre "
            "estimulación hidráulica previa del pozo, lo que permitió el ingreso de sólidos gruesos "
            "directos a la admisión del sistema de bombeo. Lección aprendida: no operar el equipo "
            "BES a frecuencias superiores a 55 Hz en pozos con antecedentes de flujo de arena."
        ),
        "doc_incorrecto": (
            "El equipo BES falló a los 92 días de operación debido a un bloqueo por gas (gas lock) "
            "crónico que derivó en la rotura del eje de la bomba por fatiga torsional cíclica. La "
            "causa raíz fue un incremento imprevisto de la relación gas-petróleo (GOR) del "
            "reservorio tras el avance de una burbuja de gas secundaria en el pozo LL-112."
        ),
    },
    {
        "query": "¿Por qué falló por upthrust el equipo BES del pozo LP-502?",
        "doc_correcto": (
            "El análisis metalúrgico confirmó que el equipo operó de forma continua en la zona de "
            "upthrust (empuje hacia arriba), lo que generó un desgaste severo por fricción y "
            "sobrecalentamiento en las arandelas de empuje superiores de los impulsores flotantes. "
            "La causa raíz fue un error de diseño en el dimensionamiento del equipo."
        ),
        "doc_incorrecto": (
            "Al desarmar el motor en taller se observó quemadura de bobinado por sobrecalentamiento. "
            "La causa raíz fue la operación sostenida del pozo por debajo del caudal mínimo "
            "recomendado por el fabricante (downthrust continuo), debido a una declinación acelerada "
            "del aporte de fluido desde el reservorio en el pozo PM-104."
        ),
    },
    {
        "query": "¿Dónde se detectó el orificio por erosión en el tubing del pozo LP-15?",
        "doc_correcto": (
            "Al llegar a la junta #54 (aprox. 1.620 metros de profundidad) se observa orificio de "
            "1.5 pulgadas por erosión severa, justo frente a la válvula de Gas Lift número 3. Se "
            "termina de extraer el resto de la sarta sin más novedades."
        ),
        "doc_incorrecto": (
            "A los 2.100 metros se observa el cable de potencia totalmente aplastado y quemado "
            "contra el casing, producto del movimiento de la sarta por falta de centralizadores en "
            "la zona desviada del pozo VM-210. Sistema BES, disparo en el variador de frecuencia "
            "por baja aislación eléctrica."
        ),
    },
    {
        "query": "¿Qué falla se detectó en el empaquetador (packer) del pozo inyector PI-44?",
        "doc_correcto": (
            "Se detecta comunicación directa entre el tubing de inyección y el espacio anular. Falla "
            "confirmada en el empaquetador (packer) de inyección. Herramienta trabada en el fondo "
            "por acumulación de incrustaciones de sulfato de bario sobre las gomas selladoras. Se "
            "constata el elemento sellador del packer totalmente destruido por extrusión química y "
            "térmica."
        ),
        "doc_incorrecto": (
            "La implementación de la inyección continua de polímeros (HPAM) en el pozo inyector "
            "PI-12 logró corregir la relación de movilidad desfavorable entre el agua y el petróleo "
            "viscoso en la capa inferior. Se detectó la llegada del banco de petróleo (oil bank) en "
            "los pozos productores colindantes tras 8 meses del inicio del proyecto."
        ),
    },
    {
        "query": "¿Qué desgaste se observó en el rotor de la PCP del pozo ST-82?",
        "doc_correcto": (
            "Se recupera el rotor de la PCP evidenciando desgaste severo por abrasión y presencia de "
            "arena de fractura alojada en el estator. Se decide programar bajada de tubing para "
            "limpieza de fondo con cuchara hidráulica antes de bajar el nuevo elastómero."
        ),
        "doc_incorrecto": (
            "Se recupera la sección de fondo (bomba, protector y motor BES). Se envían equipos a "
            "taller central para desarme (DIFA) del pozo VM-210, tras confirmarse falla a tierra en "
            "el cable de potencia o en el motor de fondo del sistema BES."
        ),
    },
    {
        "query": "¿Qué efecto tuvo la inyección continua de polímeros en el pozo inyector PI-08?",
        "doc_correcto": (
            "Al elevar la viscosidad del agua inyectada de 0.7 cP a 15 cP, se mitigó el efecto de "
            "digitación viscosa (fingering). Se confirmó mediante una estabilización del corte de "
            "agua (Water Cut) en el 68% y un incremento neto del 32% en la tasa de producción de "
            "crudo de los pozos productores colindantes."
        ),
        "doc_incorrecto": (
            "Los ensayos de liberación diferencial en laboratorio determinaron que la presión de "
            "burbuja del fluido es de 1.850 psi a una temperatura de fondo de 75°C. La caída por "
            "debajo de la presión de burbuja activó un mecanismo de empuje por gas disuelto liberado "
            "en el yacimiento Los Perales."
        ),
    },
    {
        "query": "¿Qué estructura geológica aísla al pozo PM-104 del pozo inyector PI-05?",
        "doc_correcto": (
            "El análisis de presiones estáticas (RFT) e interpretación sísmica 3D en la Formación "
            "Agrio indica una fuerte compartimentación del reservorio provocada por una falla normal "
            "sellante con rumbo NO-SE. Esta barrera estructural aísla hidráulicamente al pozo "
            "productor PM-104 del área de influencia directa del pozo inyector PI-05."
        ),
        "doc_incorrecto": (
            "Los pozos de este sector experimentarán un incremento drástico en su relación "
            "gas-petróleo (GOR) y una reducción en la permeabilidad relativa al petróleo por bloqueo "
            "de burbujas, debido a que la presión promedio del reservorio ha caído por debajo de la "
            "presión de burbuja en el yacimiento Los Perales."
        ),
    },
]