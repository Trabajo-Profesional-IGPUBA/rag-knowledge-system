import logging

from src.embeddings.evaluation import evaluate_models, generate_report

logging.basicConfig(level=logging.INFO)

from src.embeddings.corpus_loader import extract_test_texts, load_documents

documents = load_documents("data/processed")
test_texts = extract_test_texts(documents, max_docs=40)
results = evaluate_models(test_texts)
report = generate_report(results)

print(report)
with open("embeddings_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write(report)
