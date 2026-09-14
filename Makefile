IMAGE     = rag-app
TEST_IMAGE = rag-app-test
UID       := $(shell id -u)
GID       := $(shell id -g)
USER_FLAG  = --user $(UID):$(GID)
OLLAMA_BASE_URL = http://host.docker.internal:11434

.PHONY: build build-test run test lint lint-check format format-check evaluate

build:
	docker build --target runtime -t $(IMAGE) .

build-test:
	docker build --target test -t $(TEST_IMAGE) .

run: build
	docker run --rm $(USER_FLAG) -v .:/app $(IMAGE) python main.py

test: build-test
	docker run --rm $(USER_FLAG) $(TEST_IMAGE) pytest -q

lint: build-test
	docker run --rm $(USER_FLAG) -v .:/app $(IMAGE) ruff check . --fix

lint-check: build-test
	docker run --rm $(USER_FLAG) $(IMAGE) ruff check .

format: build-test
	docker run --rm $(USER_FLAG) -v .:/app $(IMAGE) black .

format-check: build-test
	docker run --rm $(USER_FLAG) $(IMAGE) black --check .

evaluate: build
	docker run --rm $(USER_FLAG) -v ./data:/app/data $(IMAGE) python run_evaluation.py

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