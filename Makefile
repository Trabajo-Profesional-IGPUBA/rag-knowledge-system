IMAGE     = rag-app
TEST_IMAGE = rag-app-test
UID       := $(shell id -u)
GID       := $(shell id -g)
USER_FLAG  = --user $(UID):$(GID)
OLLAMA_BASE_URL = http://host.docker.internal:11434
CODE_VOLUMES_FLAG = -v ./src:/app/src \
					-v ./tests:/app/tests \
					-v ./ingest_files.py:/app/ingest_files.py \
					-v ./generate_eval_file.py:/app/generate_eval_file.py \
					-v ./run_evaluation.py:/app/run_evaluation.py \
					-v ./app.py:/app/app.py \

.PHONY: build build-test run test lint lint-check format format-check evaluate

build:
	docker build --target runtime -t $(IMAGE) .

build-test:
	docker build --target test -t $(TEST_IMAGE) .

run: build
	docker run --rm $(USER_FLAG) -v ./data:/app/data -v ./logs:/app/logs $(IMAGE) python ingest_files.py

test: build-test
	docker run --rm $(USER_FLAG) $(TEST_IMAGE) pytest -q

lint: build-test
	docker run --rm -t $(USER_FLAG) $(CODE_VOLUMES_FLAG) $(TEST_IMAGE) ruff check . --fix

lint-check: build-test
	docker run --rm -t $(USER_FLAG) $(TEST_IMAGE) ruff check .

format: build-test
	docker run --rm -t $(USER_FLAG) $(CODE_VOLUMES_FLAG) $(TEST_IMAGE) black .

format-check: build-test
	docker run --rm -t $(USER_FLAG) $(TEST_IMAGE) black --check .

evaluate: build
	docker run --rm $(USER_FLAG) -v ./data:/app/data $(IMAGE) python run_evaluation.py

evaluate-llm: build
	docker run --rm $(USER_FLAG) -v ./data:/app/data $(IMAGE) python run_evaluation_llm.py

app: build
	docker run --rm \
		$(USER_FLAG) \
		-v ./data:/app/data \
		-v ./logs:/app/logs \
		-p 8501:8501 \
		-e OLLAMA_BASE_URL=$(OLLAMA_BASE_URL) \
		$(IMAGE) \
		streamlit run app.py \
			--server.address=0.0.0.0 \
			--server.port=8501