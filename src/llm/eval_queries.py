"""Dataset de consultas de referencia para evaluar el pipeline RAG con distintos modelos LLM.

Formato de `expected_keywords`: lista de grupos. Cada grupo es un str o una lista
de alternativas/sinónimos; cuenta como acierto si aparece CUALQUIERA del grupo.
Los grupos que ya aparecen en la pregunta se ignoran automáticamente al puntuar.

Las queries con `should_abstain: True` NO tienen respuesta en el corpus: lo correcto
es que el modelo se abstenga. VERIFICÁ que esos pozos/eventos realmente no existan
en tus documentos.

Corpus incluidos:

Corpus 0 (ids q*, n*): "Detalle.pdf", especificación de un caso de uso RAG para
    cuencas maduras, con documentos de ejemplo del yacimiento: perforación (EWR),
    workover y partes diarios, estimulación, integridad y corrosión, fallas de BES,
    reservorio/geología e inyección (secundaria y EOR). En español.

Corpus 1 (ids b*, nb*): "Introduction to natural gas plant LNG"
    (Hadžihafizović, 2024). En inglés. Categorías con prefijo "lng_".

En el corpus 1 las preguntas y respuestas de referencia están en español, y
los grupos de keywords incluyen variantes en español e inglés, porque los modelos
pueden citar términos originales (tags de equipos, "wet flare", etc.).
"""

from typing import Any

NO_INFO_PATTERNS = [
    "no encontré información",
    "no tengo información",
    "no se proporcionan detalles",
    "no encontré datos",
    "no dispongo de información",
    "no hay información",
    "no se encontró",
    "no se menciona",
    "no figura",
    "no cuento con información",
]

EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "¿Tuvimos problemas de pérdida de circulación en la formación Quintuco?",
        "expected_keywords": [["LCM", "cascarilla de nuez"], "PM-104"],
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
        "expected_keywords": [
            ["pesca", "herramienta de pesca"],
            "overshot",
            ["fatiga", "rotura por fatiga"],
        ],
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
            "arenamiento",
            ["presión", "psi"],
            ["estimulación", "estimulacion"],
            ["bomba", "unidad fracturadora"],
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
            ["CO2", "dióxido de carbono"],
            ["bacterias sulfato-reductoras", "BSR", "sulfato reductoras"],
            ["Water Cut", "corte de agua"],
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
        "expected_keywords": [
            ["trazador", "trazadores"],
            "PM-102",
            ["geles", "tapón de geles"],
        ],
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
            ["VSD", "variador"],
            ["aislamiento", "aislación"],
            "cable de potencia",
            "centralizadores",
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
        "expected_keywords": [
            "tubing",
            ["junta", "junta #54"],
            "metros",
            ["erosión", "erosion"],
        ],
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
            ["Run Life", "vida útil"],
            "sobrecalentamiento",
            ["declinación", "declinacion"],
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
            ["limolitas", "aislado"],
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
            ["incrustaciones", "incrustación"],
            "sulfato de bario",
            ["tracción", "traccion"],
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
    # ── Queries negativas: no hay respuesta en el corpus → el modelo debe abstenerse ──
    {
        "id": "n1",
        "query": "¿Qué pasó con la bomba de cavidad progresiva en el pozo ZZ-999?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    {
        "id": "n2",
        "query": "¿Hubo un blowout durante la perforación del pozo XQ-777?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    {
        "id": "n3",
        "query": "¿Qué resultado tuvo la inyección de polímeros en el inyector PI-99?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    {
        "id": "n4",
        "query": "¿Por qué falló el compresor del pozo GL-310 en 2022?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    # ── Libro "Introduction to natural gas plant LNG" ─────────────────────────────
    # ── Conceptos generales y seguridad ──
    {
        "id": "b1",
        "query": "¿A qué temperatura se licúa el gas natural a presión atmosférica y cuánto se reduce su volumen?",
        "expected_keywords": [
            ["-160", "−160", "–160", "160 °C", "160 ºC", "160°C"],
            ["1/600", "600"],
            ["45%", "menos de la mitad", "mitad"],
        ],
        "reference_answer": (
            "A presión atmosférica el gas natural se condensa a aproximadamente "
            "-160 °C (-256 °F) y pasa a ser LNG. Un volumen de líquido ocupa "
            "alrededor de 1/600 del volumen del gas, y el LNG pesa cerca del 45% "
            "que el agua (menos de la mitad). Es inodoro, incoloro, no corrosivo "
            "y no tóxico."
        ),
        "category": "lng_conceptos",
    },
    # ── Separación de entrada, medición y NGL ──
    {
        "id": "b2",
        "query": "¿Qué métodos se usan para medir el caudal del gas de alimentación y cuál es el menos exacto?",
        "expected_keywords": [
            ["ultrasónico", "ultrasonico", "ultrasonic"],
            ["placa orificio", "placa de orificio", "orifice"],
            "venturi",
            ["tobera", "flow nozzle", "nozzle"],
        ],
        "reference_answer": (
            "La cantidad se mide con un medidor ultrasónico, una placa orificio, "
            "un tubo Venturi o una tobera de flujo (flow nozzle). La placa "
            "orificio es la más común, de bajo costo y fácil de instalar o "
            "reemplazar, pero es la menos exacta y la que produce la mayor "
            "pérdida de presión permanente."
        ),
        "category": "lng_medición",
    },
    # ── Remoción de gas ácido (AGRU) ──
    {
        "id": "b3",
        "query": "¿Qué solvente se usa para remover CO2 y H2S y por qué se eligió?",
        "expected_keywords": [
            [
                "DGA",
                "diglicolamina",
                "diglycolamine",
                "diglicol amina",
                "diglycol amine",
            ],
            ["amina primaria", "primary amine"],
            ["COS", "mercaptanos", "mercaptans"],
            ["50%", "50 %"],
        ],
        "reference_answer": (
            "Se usa DGA (diglicolamina), una amina primaria, en solución al 50% "
            "en peso con un caudal de 116 m³/h. Se eligió por su eficiencia y "
            "capacidad para bajar el CO2 a menos de 100 ppmv a un costo "
            "competitivo, y además remueve COS y algunos mercaptanos."
        ),
        "category": "lng_gas_ácido",
    },
    # ── Deshidratación y mercurio ──
    {
        "id": "b4",
        "query": "¿Cuántos secadores de tamiz molecular tiene la unidad de deshidratación y cuánto dura cada etapa de la regeneración?",
        "expected_keywords": [
            ["tres", "three", "3 secadores"],
            ["4.5", "4,5"],
            ["287.8", "287,8", "287"],
            ["3 horas", "3 h", "tres horas", "3 hrs"],
        ],
        "reference_answer": (
            "Hay tres secadores de tamiz molecular (V-1302 A/B/C): dos en servicio "
            "y uno en regeneración o en espera. La regeneración térmica comienza "
            "con un paso de calentamiento de 4,5 horas con gas seco calentado "
            "hasta 287,8 °C, seguido de un paso de enfriamiento de 3 horas con "
            "gas sin calentar. La unidad reduce el agua a menos de 1 ppmv."
        ),
        "category": "lng_deshidratación",
    },
    # ── Refrigeración en cascada y licuefacción ──
    {
        "id": "b5",
        "query": "¿Cómo funciona la cascada de refrigeración de la planta y a qué temperatura llega el gas?",
        "expected_keywords": [
            ["Phillips"],
            "propano",
            "etileno",
            "metano",
            ["-161", "−161", "–161"],
        ],
        "reference_answer": (
            "La planta usa el proceso Phillips Cascade con tres circuitos de "
            "refrigeración: propano, etileno y metano, que enfrían el gas por "
            "etapas hasta -161 °C. El propano enfría la alimentación antes y "
            "después de la deshidratación; el gas de la última etapa de propano "
            "junto con una corriente de reciclo de metano se enfría y condensa en "
            "el circuito de etileno; y el condensado entra a un ciclo abierto de "
            "metano que produce el LNG, enviado a tanques. El boil-off de los "
            "tanques se comprime y vuelve al reciclo."
        ),
        "category": "lng_refrigeración",
    },
    # ── Pesados y NGL ──
    {
        "id": "b6",
        "query": "¿Para qué sirve la columna de remoción de pesados y qué corrientes maneja?",
        "expected_keywords": [
            ["aromáticos", "aromaticos", "aromatics"],
            ["congel", "freez"],
            ["reflujo", "reflux"],
            ["-76.4", "−76.4", "–76.4", "-76,4", "40.9", "40,9"],
        ],
        "reference_answer": (
            "La columna de remoción de pesados (HRC, V-1701) es una columna "
            "empacada que separa etano y componentes más pesados (incluidos "
            "aromáticos) para evitar congelamiento y bloqueos en la sección de "
            "licuefacción. La mayor parte del gas se enfría a -76,4 °C y 40,9 bar "
            "antes de entrar a la zona media; el resto, enfriado en el "
            "intercambiador E-1704, entra por el fondo. El reflujo es metano "
            "totalmente condensado desde E-1504A/B. Los fondos pasan a la columna "
            "de recuperación de NGL (V-1702) que separa los pesados (C5+) para "
            "enviarlos al almacenamiento de NGL."
        ),
        "category": "lng_pesados_NGL",
    },
    # ── Antorchas, combustible, refrigerantes ──
    {
        "id": "b7",
        "query": "¿Cuántos sistemas de antorcha tiene la planta y qué maneja cada uno?",
        "expected_keywords": [
            ["húmeda", "humeda", "wet"],
            ["seca", "dry"],
            ["marina", "marine"],
            ["criogénic", "criogenic", "cryogenic"],
        ],
        "reference_answer": (
            "Hay tres antorchas separadas. La antorcha de gas húmedo (wet flare, "
            "K-1901, de acero al carbono) maneja hidrocarburos tibios saturados con "
            "agua o con líquidos libres. La antorcha de gas seco (dry flare, "
            "K-1902, de acero inoxidable) maneja vapores y líquidos de "
            "hidrocarburos criogénicos. La antorcha marina (K-1903) está cerca "
            "del área de carga de buques y maneja los vapores de LNG flasheado "
            "durante la carga o ante disparo de los compresores de BOG. Todas "
            "tienen ignición electrónica por chispa."
        ),
        "category": "lng_antorchas",
    },
    # ── Almacenamiento y carga de LNG ──
    {
        "id": "b8",
        "query": "¿Cuántos tanques de LNG hay, qué capacidad tienen y de qué material es el tanque interno?",
        "expected_keywords": [
            ["140,000", "140.000", "140 000", "140000"],
            ["dos", "two", "T-2401"],
            ["9% níquel", "9% niquel", "9% nickel", "níquel", "nickel"],
            [
                "conexiones superiores",
                "top connections",
                "parte superior",
                "llenado por arriba",
                "top-fill",
                "top filling",
            ],
        ],
        "reference_answer": (
            "Hay dos tanques de LNG (T-2401A/B), cada uno de 140.000 m³, con "
            "tanque interno de acero al 9% de níquel y el espacio hasta la pared "
            "exterior relleno de aislación. Sólo tienen conexiones por la parte "
            "superior (ninguna cañería ni instrumento penetra lados o fondo) y se "
            "llenan por arriba mediante una placa deflectora (splash plate). Cada "
            "tanque tiene cuatro bombas de carga sumergidas."
        ),
        "category": "lng_almacenamiento",
    },
    # ── Queries negativas del libro LNG (Skikda y CAPEX no aparecen en el libro) ──
    {
        "id": "nb1",
        "query": "¿Qué pasó en el accidente de la planta de LNG de Skikda en 2004?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    {
        "id": "nb2",
        "query": "¿Cuál fue el CAPEX total de construcción de la planta de LNG?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
]
