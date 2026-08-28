import logging
from src.embeddings.evaluation import evaluate_models, generate_report

logging.basicConfig(level=logging.INFO)

from src.embeddings.corpus_loader import cargar_documentos, extraer_test_texts

documentos = cargar_documentos("data/processed")
textos_de_prueba = extraer_test_texts(documentos, max_docs=40)
resultados = evaluate_models(textos_de_prueba)
reporte = generate_report(resultados)

print(reporte)
with open("reporte_evaluacion_embeddings.md", "w", encoding="utf-8") as f:
    f.write(reporte)