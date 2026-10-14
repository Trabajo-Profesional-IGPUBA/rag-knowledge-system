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

Corpus 2 (ids lng*, lngn*): "Handbook of Liquefied Natural Gas"
    (Mokhatab, Mak, Valappil, Wood; Elsevier, 2014). En inglés.
    Categorías con prefijo "lng_".

En los corpus 1 y 2 las preguntas y respuestas de referencia están en español, y
los grupos de keywords incluyen variantes en español e inglés, porque los modelos
pueden citar términos originales (tags de equipos, "wet flare", etc.).

Con los corpus 1 y 2 en el mismo índice, las preguntas del corpus 1 hablan de
"la planta" y las del corpus 2 usan formulaciones genéricas, para evitar ambigüedad.
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
    # Corpus 0: Detalle.pdf
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
    # ── Corpus 1: Libro "Introduction to natural gas plant LNG" ─────────────────────────────
    # ── Conceptos generales y seguridad ──
    {
        "id": "b1",
        "query": "¿Qué capacidad nominal de producción tiene la planta y a qué velocidad máxima se carga un buque?",
        "expected_keywords": [
            ["3.6", "3,6"],
            ["340"],
            ["10,080", "10.080", "10080"],
            ["millones", "million", "MTPA"],
        ],
        "reference_answer": (
            "La planta (ELNG, tren 1) es una instalación nueva y autosuficiente "
            "diseñada para producir nominalmente 3,6 millones de toneladas "
            "métricas por año de LNG (con 340 días de operación). En modo de "
            "carga, todas las bombas de los dos tanques cargan un buque a una "
            "velocidad máxima de 10.080 m³/h."
        ),
        "category": "lng_conceptos",
    },
    {
        "id": "b2",
        "query": "¿Qué métodos se usan para medir el caudal del gas de alimentación de la planta y cuál es el menos exacto?",
        "expected_keywords": [
            ["ultrasónico", "ultrasonico", "ultrasonic"],
            ["placa orificio", "placa de orificio", "orifice"],
            "venturi",
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
    {
        "id": "b3",
        "query": "¿Qué solvente usa la planta para remover CO2 y H2S y por qué se eligió?",
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
    {
        "id": "b4",
        "query": "¿Cuántos secadores de tamiz molecular tiene la unidad de deshidratación de la planta y cuánto dura cada etapa de la regeneración?",
        "expected_keywords": [
            ["tres", "three", "3 secadores"],
            ["4.5", "4,5"],
            ["287.8", "287,8", "287"],
            ["3 horas", "3 h", "tres horas", "3 hrs", "3.0", "3,0"],
        ],
        "reference_answer": (
            "Hay tres secadores de tamiz molecular (V-1302 A/B/C): dos en servicio "
            "y uno en regeneración o en espera. La regeneración térmica comienza "
            "con un paso de calentamiento de 4,5 horas con gas seco calentado "
            "hasta 287,8 °C, seguido de un paso de enfriamiento de 3 horas con "
            "gas sin calentar (el ciclo completo de cada lecho es: adsorción 16 h, "
            "calentamiento 4,5 h, enfriamiento 3,0 h y espera 0,5 h). La unidad "
            "reduce el agua a menos de 1 ppmv."
        ),
        "category": "lng_deshidratación",
    },
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
    {
        "id": "b6",
        "query": "¿Para qué sirve la columna de remoción de pesados de la planta y qué corrientes maneja?",
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
    {
        "id": "b8",
        "query": "¿A qué presiones y temperaturas opera el circuito de propano de la planta en sus tres niveles?",
        "expected_keywords": [
            ["5.47", "5,47"],
            ["2.72", "2,72"],
            ["0.39", "0,39"],
            [
                "10.3",
                "10,3",
                "-7.8",
                "-7,8",
                "−7.8",
                "–7.8",
                "-34.5",
                "-34,5",
                "−34.5",
                "–34.5",
            ],
        ],
        "reference_answer": (
            "El propano refrigerante opera en tres niveles de presión: alta "
            "etapa a 5,47 bar y 10,3 °C, etapa intermedia a 2,72 bar y -7,8 °C, "
            "y baja etapa a 0,39 bar y -34,5 °C."
        ),
        "category": "lng_refrigeración",
    },
    # Negativas del corpus 1.
    {
        "id": "nb1",
        "query": "¿Qué función cumple la columna V-9902 de la planta?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    {
        "id": "nb2",
        "query": "¿Cuántas bombas de carga tiene el tanque T-2403 de la planta?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    #  Corpus 2: Handbook of Liquefied Natural Gas (Mokhatab et al., 2014) ══════════
    {
        "id": "lng1",
        "query": "¿Qué es el rollover en un tanque de GNL y cómo se previene?",
        "expected_keywords": [
            ["estratificación", "estratificacion", "stratification"],
            ["densidad", "density"],
            ["nitrógeno", "nitrogeno", "nitrogen"],
            ["recirculación", "recirculacion", "recircul", "mezcla", "mixing"],
        ],
        "reference_answer": (
            "El rollover ocurre cuando el GNL se estratifica en el tanque en dos "
            "capas de distinta densidad durante un período prolongado y con "
            "mezcla insuficiente. La capa superior pierde componentes livianos "
            "(weathering) y se vuelve más densa, hasta que se hunde y se mezcla "
            "bruscamente con la capa inferior, que está sobrecalentada y libera "
            "gran cantidad de vapor; esto puede causar un aumento súbito de "
            "presión y daño estructural, porque las válvulas de alivio no están "
            "dimensionadas para ese evento. Se origina por estratificación "
            "inducida por llenado (carga más o menos densa que el remanente "
            "cargada por la línea equivocada) o por nitrógeno en concentraciones "
            "mayores al 1%. Se previene monitoreando presión, densidad y "
            "temperatura a lo largo de la columna de líquido, manteniendo "
            "mezcla o recirculación y usando múltiples puntos de carga a "
            "distintos niveles del tanque."
        ),
        "category": "lng_almacenamiento",
    },
    {
        "id": "lng2",
        "query": "¿Qué tipos de tanques de almacenamiento se usan en terminales de GNL en tierra y cuál es el más común actualmente?",
        "expected_keywords": [
            ["contención total", "contencion total", "full containment"],
            ["contención simple", "contencion simple", "single containment"],
            ["contención doble", "contencion doble", "double containment"],
            ["10 y 20", "10 a 20", "10-20", "10 to 20"],
        ],
        "reference_answer": (
            "Se usan tres tipos de tanques: de contención simple, de contención "
            "doble y de contención total. El de contención simple (cilindro "
            "interno de acero con 9% de níquel, tanque externo de acero al "
            "carbono con perlita y un muro o dique externo) es la opción más "
            "económica, pero requiere más terreno y tiene primas de seguro más "
            "altas. El de contención total es el más usado: la mayoría de los "
            "tanques construidos en los 10 años previos a la publicación del "
            "libro son de este tipo. Cuesta entre 10 y 20% más que el de "
            "contención simple, pero ofrece una capa adicional de seguridad "
            "frente a fuego, explosiones e impactos externos, ya que el "
            "contenedor secundario es hermético a líquido y vapor."
        ),
        "category": "lng_almacenamiento",
    },
    {
        "id": "lng3",
        "query": "¿Qué tipos de vaporizadores se usan en las terminales de regasificación y en qué proporción?",
        "expected_keywords": [
            ["ORV", "open rack", "bastidor abierto", "rack abierto"],
            [
                "SCV",
                "combustión sumergida",
                "combustion sumergida",
                "submerged combustion",
            ],
            ["70%", "70 %", "70 por ciento"],
            ["20%", "20 %", "20 por ciento"],
        ],
        "reference_answer": (
            "Las terminales de regasificación de carga base usan principalmente "
            "dos tipos de vaporizadores probados: los de bastidor abierto "
            "(Open Rack Vaporizer, ORV), en aproximadamente 70% de los casos, y "
            "los de combustión sumergida (Submerged Combustion Vaporizer, SCV), "
            "en alrededor de 20%. También se han usado vaporizadores de aire "
            "ambiente (AAV), de casco y tubos (STV) y de fluido intermedio "
            "(IFV). Los ORV usan agua (comúnmente de mar) como fuente de calor "
            "sobre tubos aletados de aleación de aluminio."
        ),
        "category": "lng_regasificación",
    },
    {
        "id": "lng4",
        "query": "¿Cuánto combustible consume un vaporizador de combustión sumergida y por qué se usa solo en ciertos casos?",
        "expected_keywords": [
            ["1.5%", "1,5%", "1.5 %", "1,5 %"],
            ["fuente de calor", "heat source", "calor gratuito", "free heat"],
            ["98%", "98 %"],
        ],
        "reference_answer": (
            "Un SCV requiere aproximadamente 1,5% del GNL vaporizado total como "
            "combustible, lo que agrega un costo operativo significativo; por "
            "eso se usa solo cuando no hay otra fuente de calor gratuita "
            "disponible. Puede diseñarse para aprovechar boil-off gas. Es "
            "compacto, tiene una eficiencia térmica superior al 98%, maneja bien "
            "las fluctuaciones de carga y una unidad puede llegar a unos 200 "
            "MMSCFD, aunque sus controles son más complejos que los de un ORV."
        ),
        "category": "lng_regasificación",
    },
    {
        "id": "lng8",
        "query": "¿A partir de qué distancias el GNL resulta más competitivo que el transporte por gasoducto y qué proporción del gas comerciado internacionalmente se exporta por cada medio?",
        "expected_keywords": [
            "700",
            ["2.200", "2,200", "2200"],
            ["70%", "70 %", "70 por ciento", "70 percent"],
            ["30%", "30 %", "30 por ciento", "30 percent"],
        ],
        "reference_answer": (
            "Para distancias cortas los gasoductos suelen ser más económicos; el "
            "GNL se vuelve competitivo en rutas largas, sobre todo si cruzan "
            "océanos, porque los gasoductos submarinos son prohibitivos. Para "
            "gas offshore, el GNL puede competir cuando el gasoducto offshore "
            "tendría menos de 700 millas; para gasoductos onshore el punto de "
            "equilibrio es de unas 2.200 millas. Cerca de 70% del gas comerciado "
            "internacionalmente se exporta por gasoducto y el 30% restante como "
            "GNL."
        ),
        "category": "lng_economía",
    },
    {
        "id": "lng10",
        "query": "¿Por qué se limita el contenido de nitrógeno en el GNL y cómo se remueve?",
        "expected_keywords": [
            ["1 mol", "1 %", "1%", "1 mole", "1 por ciento"],
            ["rollover", "estratificación", "estratificacion", "stratification"],
            ["flash", "flasheo", "end-flash"],
            ["gas combustible", "fuel gas", "boil-off"],
        ],
        "reference_answer": (
            "El contenido de nitrógeno del GNL se limita típicamente a cerca de "
            "1% molar. Un nitrógeno bajo evita temperaturas de licuefacción más "
            "bajas y mayor consumo de potencia de refrigeración, reduce el "
            "nitrógeno en el boil-off para que pueda usarse como gas combustible "
            "y reduce el riesgo de rollover en los tanques de las terminales. "
            "Se remueve con un flash final (end-flash): el GNL se expande en un "
            "tambor a presión cercana a la de almacenamiento y el nitrógeno, "
            "más liviano, se separa como vapor y se recupera como gas "
            "combustible. Este esquema es adecuado para gases de alimentación "
            "con hasta unos 2% molar de nitrógeno; con más nitrógeno se "
            "requiere una etapa adicional de fraccionamiento."
        ),
        "category": "lng_licuefacción",
    },
    # Negativas del Corpus 2
    {
        "id": "lngn1",
        "query": "¿Cómo afectó el accidente de Fukushima a la demanda de GNL según el libro?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
    {
        "id": "lngn2",
        "query": "¿Qué dice el libro sobre el gasoducto Nord Stream?",
        "expected_keywords": [],
        "should_abstain": True,
        "category": "sin_respuesta",
    },
]
