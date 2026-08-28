import logging
from src.embeddings.evaluation import evaluate_models, generate_report

logging.basicConfig(level=logging.INFO)

textos_de_prueba = [
    "Requisitos y pasos para obtener o renovar la licencia de conducir.",
    "Documentación necesaria para el registro en el padrón de votantes.",
    "Canal de reclamos por infraestructura vial y baches.",
    "Trámite y requisitos para obtener el certificado único de discapacidad.",
    "Plataforma de pago digital de tasas y contribuciones municipales.",
    "Sistema de turnos online para trámites en el registro civil.",
    "Pasos y documentación para tramitar la habilitación comercial de un local.",
    "Procedimiento para realizar una denuncia por ruidos molestos.",
    "Requisitos para solicitar la exención de tasas municipales para jubilados.",
    "Trámite de baja de vehículo ante el registro correspondiente.",
    "Requisitos e inscripción online a jardines maternales municipales.",
    "Documentación y turno necesarios para contraer matrimonio civil.",
    "Canal para reportar fallas en el alumbrado público de la vía pública.",
    "Procedimiento para solicitar audiencia con autoridades municipales.",
    "Requisitos para tramitar permiso de uso del espacio público para eventos.",
    "Trámite para obtener el certificado de residencia municipal.",
    "Requisitos para acceder al subsidio municipal por desempleo.",
    "Requisitos e inscripción para obtener el boleto de transporte estudiantil.",
    "Procedimiento para solicitar la poda de árboles en el espacio público.",
    "Trámite y turno para la renovación del DNI en oficinas del registro civil.",
    "Requisitos y documentación para iniciar el trámite de jubilación municipal.",
    "Requisitos para inscribirse como expositor en ferias itinerantes municipales.",
    "Horarios de atención y servicios de la biblioteca pública municipal.",
    "Procedimiento para solicitar traslados en ambulancia del sistema de salud municipal.",
    "Acceso y consulta del boletín oficial municipal con ordenanzas vigentes.",
    "Procedimiento para presentar un reclamo en la oficina de defensa del consumidor.",
    "Requisitos para tramitar el permiso de obra de construcción de vivienda unifamiliar.",
    "Requisitos y turno para tramitar la libreta sanitaria municipal.",
    "Procedimiento para solicitar el retiro municipal de residuos voluminosos.",
    "Requisitos para el primer empadronamiento de votantes que alcanzan la mayoría de edad.",
]

resultados = evaluate_models(textos_de_prueba)
reporte = generate_report(resultados)

print(reporte)
with open("reporte_evaluacion_embeddings.md", "w", encoding="utf-8") as f:
    f.write(reporte)