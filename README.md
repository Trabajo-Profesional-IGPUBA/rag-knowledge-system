## Instalación

```bash
git clone https://github.com/tu-usuario/rag-knowledge-system.git
cd rag-knowledge-system
```

## Desarrollo Local

### Instalacion de paquetes y dependencias

```bash
apt-get install -y --no-install-recommends libxcb1 libxrender1 libxext6 libgl1 libglib2.0-0 libsm6
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Docker

### Construir las imágenes

```bash
make build        # imagen de producción
make build-test   # imagen de desarrollo y tests
```

### Levantar la interfaz web

```bash
make app
```

Disponible en `http://localhost:8501`. Requiere Ollama corriendo en el host.

### Ejecutar la ingesta

```bash
make run
```

### Ejecutar los tests

```bash
make test
```

### Formatear el código

```bash
make format         # aplica formato con black
make lint           # aplica fixes con ruff
```

### Verificar el estilo del código

```bash
make format-check   # solo reporta problemas de formato
make lint-check     # solo reporta errores de lint
```

### Evaluación de modelos de embeddings

```bash
make evaluate
```

### Evaluación de LLM

```bash
make evaluate-llm
```

Para personalizarla, pasa los parámetros con `ARGS`:

```bash
make evaluate-llm ARGS="--models <modelos> --judge-model <juez> --n-runs <n> --min-faithfulness <min> --latency-penalty <penalización> --persist-dir <carpeta> --output <ruta>"
```

Todos los parámetros son opcionales:

- `--models`: modelos de Ollama a comparar, separados por espacio.
- `--judge-model`: modelo juez, distinto de los evaluados e idealmente más grande. Sin él no se mide corrección ni fidelidad.
- `--n-runs`: repeticiones por consulta, para promediar la variación (por defecto 3; usa 1 para pruebas rápidas).
- `--min-faithfulness`: fidelidad mínima exigida (escala 1-5). Requiere `--judge-model`.
- `--latency-penalty`: puntos de calidad (0-1) que se exigen por cada vez que un modelo tarda el doble que el más rápido. Más alto penaliza más a los modelos lentos.
- `--persist-dir`: carpeta de la base vectorial (por defecto `data/vectorstore`).
- `--output`: ruta del reporte JSON (por defecto `eval_results/report.json`).

## CI

El CI verifica automáticamente lint y formato en cada push o pull request, y bloquea el merge si no se cumplen. Los tests corren con cobertura y el reporte se sube como artefacto del workflow.

### Hook de pre-commit (recomendado)

Instalar dependencias de desarrollo (una sola vez):

```bash
pip install pre-commit
pre-commit install
```

A partir de ahí, cada `git commit` corre lint y formato automáticamente sobre los archivos modificados. Para correrlo manualmente:

```bash
pre-commit run --all-files
```

## Equipo

| Integrante | Padrón |
|---|---|
| Finci, Dante Alejandro | 108456 |
| Lanfranco, Tomás | 110883 |
| Rodriguez, Franco Ezequiel | 102815 |
| Urbano, Sol Guadalupe | 109525 |

**Tutor:** Dr. Hernán Daniel Merlino  
**Co-tutora:** Dra. Gabriela Beatriz Savioli  
**Instituto:** IGPUBA — Facultad de Ingeniería, UBA