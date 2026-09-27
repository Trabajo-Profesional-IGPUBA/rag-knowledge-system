FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxrender1 libxext6 libgl1 libglib2.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/cache/huggingface

# Inicializar modelos
RUN python - <<'EOF'
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

DocumentConverter()
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EOF

RUN chmod -R 777 /cache && chmod -R 777 /usr/local/lib/python3.12/site-packages/rapidocr

FROM builder AS test

COPY src/ ./src/
COPY tests/ ./tests/
COPY ingest_files.py .
COPY app.py .
COPY run_evaluation.py .
COPY run_evaluation_llm.py .
COPY generate_eval_file.py .
COPY pyproject.toml .
RUN chmod 777 /app


CMD ["pytest"]


FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.12/site-packages \
    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache

RUN mkdir /.streamlit && chmod 777 /.streamlit

ENV HF_HOME=/cache/huggingface

WORKDIR /app
RUN chmod 777 /app
COPY src/ ./src/
COPY ingest_files.py .
COPY app.py .
COPY run_evaluation.py .
COPY run_evaluation_llm.py .
COPY generate_eval_file.py .

CMD ["python", "ingest_files.py"]