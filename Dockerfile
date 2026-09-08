FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxrender1 libxext6 libgl1 libglib2.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

WORKDIR /app

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

RUN chmod -R 777 /cache

FROM builder AS test

COPY src/ ./src/
COPY tests/ ./tests/
COPY main.py .
COPY app.py .
COPY run_evaluation.py .
COPY generate_eval_file.py .


CMD ["pytest"]


FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.12/site-packages \
    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /cache /cache

ENV HF_HOME=/cache/huggingface

WORKDIR /app
COPY src/ ./src/
COPY main.py .
COPY app.py .
COPY run_evaluation.py .
COPY generate_eval_file.py .

CMD ["python", "main.py"]