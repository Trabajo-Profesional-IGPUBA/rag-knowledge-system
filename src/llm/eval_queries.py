"""Dataset de consultas de referencia para evaluar el pipeline RAG con distintos modelos LLM."""

from typing import Any

NO_INFO_PATTERNS = [
    "no encontré información",
    "no tengo información",
    "no se proporcionan detalles",
    "no encontré datos",
    "no dispongo de información",
]

EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "¿Tuvimos problemas de pérdida de circulación en la formación Quintuco?",
        "expected_keywords": ["pérdida de circulación", "Quintuco", "LCM", "PM-104"],
        "reference_answer": (
            "Sí, en 2012 el pozo PM-104 tuvo una pérdida de circulación severa "
            "de aproximadamente 15 m³/h en la formación Quintuco, a los 2.450 "
            "metros mdf. Se solucionó bombeando píldoras de LCM (cascarilla de "
            "nuez y carbonato de calcio), y se recomendó no superar un peso de "
            "lodo de 1.15 g/cm³ en pozos colindantes."
        ),
        "category": "perforación",
    },
    {
        "id": "q2",
        "query": "¿Qué pasó con la sarta de varillas en el pozo LL-205?",
        "expected_keywords": ["varillas", "pesca", "overshot", "LL-205", "fatiga"],
        "reference_answer": (
            "En el pozo LL-205 (2019) se produjo un desprendimiento de varillas "
            "a los 1.820 metros por rotura de fatiga en el cuello de una varilla "
            'de 7/8" grado D. Se recuperó con una herramienta de pesca tipo '
            'overshot de 2 3/8", tras limpiar parafina con gasoil caliente. Se '
            "recomendó instalar centralizadores de alta temperatura en el tramo "
            "desviado."
        ),
        "category": "workover",
    },
    {
        "id": "q3",
        "query": "¿Qué causó el screen-out durante la fractura hidráulica en CH-45?",
        "expected_keywords": [
            "screen-out",
            "arenamiento",
            "presión",
            "CH-45",
            "estimulación",
        ],
        "reference_answer": (
            "En el pozo CH-45 (2021), durante la etapa 4 de estimulación "
            "hidráulica en la formación Agrio, un incremento repentino de "
            "presión (de 6.200 a 8.500 psi) al ingresar arena mesh 20/40 "
            "provocó un screen-out (arenamiento) prematuro, originado por una "
            "falla mecánica en una bomba de la unidad fracturadora."
        ),
        "category": "estimulación",
    },
    {
        "id": "q4",
        "query": "¿Qué mecanismo de corrosión afectó al tubing del pozo YPF-X2?",
        "expected_keywords": [
            "corrosión",
            "CO2",
            "bacterias sulfato-reductoras",
            "tubing",
            "Water Cut",
        ],
        "reference_answer": (
            "El tubing del pozo YPF-X2 (2017) sufrió corrosión por flujo "
            "bifásico con alta concentración de CO2 disuelto y presencia de "
            "bacterias sulfato-reductoras (BSR), agravada por un Water Cut "
            "superior al 85%, con pérdida de sección de hasta el 65% del "
            "espesor de pared entre 1.100 y 1.250 metros."
        ),
        "category": "integridad",
    },
    {
        "id": "q5",
        "query": "¿Hubo canalización preferencial entre el inyector PI-08 y algún pozo productor?",
        "expected_keywords": ["trazador", "canalización", "PI-08", "PM-102", "barrido"],
        "reference_answer": (
            "Sí, un ensayo de trazador en el pozo inyector PI-08 (2024) mostró "
            "llegada al pozo productor PM-102 en solo 12 días (vs. 45 días "
            "históricos), confirmando canalización preferencial, con el corte "
            "de agua del PM-102 subiendo del 50% al 94%. Se decidió suspender "
            "la inyección y tratar con un tapón de geles."
        ),
        "category": "reservorio_inyección",
    },
    {
        "id": "q6",
        "query": "¿Tuvimos problemas con el cable de la BES?",
        "expected_keywords": [
            "VSD",
            "aislamiento",
            "caja de venteo",
            "BES",
            "cable de potencia",
        ],
        "reference_answer": (
            "Sí, en el pozo VM-210 (2023) el sistema BES se detuvo por baja "
            "aislación eléctrica (<0.1 Megaohms) detectada por el VSD. Al "
            "extraer la sarta se encontró el cable de potencia aplastado y "
            "quemado contra el casing a 2.100 metros, por falta de "
            "centralizadores en la zona desviada."
        ),
        "category": "BES",
    },
    {
        "id": "q7",
        "query": "¿A qué profundidad falló la tubería en el pozo LP-15?",
        "expected_keywords": ["tubing", "junta", "metros", "erosión", "LP-15"],
        "reference_answer": (
            'En el pozo LP-15 (2025) el tubing de 2 7/8" falló en la junta #54, '
            "a aproximadamente 1.620 metros de profundidad, con un orificio de "
            "1.5 pulgadas por erosión severa, justo frente a la válvula de Gas "
            "Lift número 3."
        ),
        "category": "workover",
    },
    {
        "id": "q8",
        "query": "¿Qué problemas tuvimos con las BES en este yacimiento por baja tasa de flujo?",
        "expected_keywords": [
            "downthrust",
            "Run Life",
            "sobrecalentamiento",
            "declinación",
            "BES",
        ],
        "reference_answer": (
            "En el pozo PM-104 (2021), el equipo BES falló tras solo 45 días de "
            "Run Life por sobrecalentamiento del bobinado, con downthrust "
            "continuo causado por la declinación acelerada del aporte de "
            "fluido del reservorio, sin que el operador lo advirtiera a tiempo "
            "por falla del sensor de fondo."
        ),
        "category": "BES",
    },
    {
        "id": "q9",
        "query": "¿Por qué el pozo PM-104 está produciendo más gas si no se tocó el estrangulador?",
        "expected_keywords": [
            "presión de burbuja",
            "gas disuelto",
            "PM-104",
            "GOR",
            "PVT",
        ],
        "reference_answer": (
            "El aumento de gas en el PM-104 se debe a una dinámica de "
            "reservorio: la presión del sector cayó a 1.650 psi, por debajo de "
            "la presión de burbuja de 1.850 psi, liberando gas disuelto y "
            "activando un mecanismo de empuje por gas, lo que incrementa el "
            "GOR, según el Informe de Estudios PVT del Yacimiento Los Perales "
            "(2024)."
        ),
        "category": "reservorio",
    },
    {
        "id": "q10",
        "query": "¿Por qué el pozo PM-108 tiene baja productividad crónica?",
        "expected_keywords": [
            "facies",
            "arcillosa",
            "canales fluviales",
            "PM-108",
            "conectividad",
        ],
        "reference_answer": (
            "La baja productividad crónica del pozo PM-108 no se debe a daño "
            "mecánico de la formación, sino a que el pozo interceptó una "
            "facies de llanura de inundación arcillosa (limolitas "
            "impermeables), quedando aislado del sistema principal de canales "
            "fluviales del yacimiento, según el estudio sedimentológico de la "
            "Formación Challacó."
        ),
        "category": "geología",
    },
    {
        "id": "q11",
        "query": "¿Qué falla mecánica ocurrió en el packer del pozo inyector PI-44?",
        "expected_keywords": [
            "packer",
            "incrustaciones",
            "sulfato de bario",
            "PI-44",
            "tracción",
        ],
        "reference_answer": (
            "En el pozo inyector PI-44 (2026) se detectó una falla en el "
            "packer (modelo Baker AD-1, a 1.950 metros) por comunicación "
            "directa entre el tubing de inyección y el espacio anular. La "
            "herramienta quedó trabada por incrustaciones de sulfato de bario "
            "sobre las gomas selladoras, y se liberó tras maniobras de "
            "tracción de hasta 30.000 lbs; el elemento sellador quedó "
            "destruido por extrusión química y térmica."
        ),
        "category": "workover",
    },
]
