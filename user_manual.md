# Manual de Usuario — Sistema RAG IGPUBA

## 1. Introducción

El Sistema RAG IGPUBA permite consultar documentación técnica de pozos petroleros en lenguaje natural. El usuario escribe preguntas como *"¿Qué problemas de circulación hubo en el pozo PM-104?"* y el sistema busca en los documentos indexados y genera una respuesta basada en ellos.

El sistema tiene dos operaciones principales:

1. **Ingesta**: procesar PDFs técnicos y cargarlos en la base de datos vectorial.
2. **Consulta**: interactuar con el sistema a través de una interfaz de chat web.

Este manual está dirigido a técnicos o ingenieros que necesiten operar el sistema. No se requiere conocimiento de programación.

## 2. Requisitos previos

Antes de instalar el sistema, verificá que tenés lo siguiente:

| Requisito | Versión mínima | Para qué se usa |
| --- | --- | --- |
| [Docker](https://docs.docker.com/get-docker/) | 24.x | Correr el sistema en contenedores |
| [Git](https://git-scm.com/) | 2.x | Clonar el repositorio |
| [Ollama](https://ollama.com) | 0.3.x | Correr el modelo de lenguaje localmente |

### Modelos requeridos en Ollama

Una vez instalado Ollama, descargá el modelo que usa la app:

```bash
ollama pull qwen3:8b
```

Para la evaluación comparativa de LLMs (opcional), podés agregar otros modelos:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### Recursos de hardware recomendados

- RAM: 16 GB o más
- Espacio en disco: 20 GB libres (modelos + vectorstore)
- CPU: 4 núcleos o más (el procesamiento de PDFs es intensivo)

## 3. Instalación y configuración

### Clonar el repositorio

```bash
git clone https://github.com/Trabajo-Profesional-IGPUBA/rag-knowledge-system.git
cd rag-knowledge-system
```

### Construir las imágenes Docker

```bash
make build
```

Este comando construye la imagen de producción. La primera vez puede tardar varios minutos porque descarga las dependencias y los modelos de OCR.

Para construir la imagen de desarrollo y tests:

```bash
make build-test
```

### Variables de entorno

Copiá el archivo de ejemplo y ajustá si es necesario:

```bash
cp .env.example .env
```

La única variable relevante en la mayoría de los casos es:

| Variable | Valor por defecto | Descripción |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | URL del servidor Ollama |

Si Ollama corre en otra máquina o puerto, modificá esa variable en `.env`.

### Verificar que Ollama está corriendo

Antes de usar el sistema, Ollama tiene que estar activo:

```bash
ollama serve
```

Dejá este proceso corriendo en una terminal aparte.

## 4. Ingesta de documentos

La ingesta procesa los PDFs técnicos, extrae su texto, genera embeddings y los carga en la base de datos vectorial. Este paso es necesario antes de poder hacer consultas.

### Estructura de carpetas

Los documentos deben colocarse en `data/raw/`, organizados por tipo:

```
data/raw/
├── ewrs/              # Informes finales de perforación
├── workovers/         # Informes de workover
├── partes_diarios/    # Partes diarios de operaciones
├── estimulaciones/    # Informes de estimulación y fractura
├── fallas_bes/        # Análisis de fallas de BES
├── fallas_integridad/ # Fallas de integridad
├── reservorio/        # Estudios de reservorio y geología
├── sedimentologia/    # Estudios sedimentológicos
└── libros/            # Bibliografía técnica de referencia (manuales, libros)
```

El tipo de carpeta determina cómo se clasifica el documento en el sistema. En la carpeta `libros/` no se extraen identificadores de pozo, ya que se trata de material teórico.

### Ejecutar la ingesta

```bash
make run
```

Este comando procesa todos los PDFs en `data/raw/` y sus subcarpetas.

Para procesar un archivo o carpeta específica:

```bash
docker run --rm -v ./data:/app/data rag-app python ingest_files.py data/raw/ewrs/
```

Para controlar la cantidad de procesos paralelos (por defecto usa hasta 4):

```bash
docker run --rm -v ./data:/app/data rag-app python ingest_files.py --max-workers 2
```

### Verificar que la ingesta funcionó

Al finalizar, el sistema imprime un resumen en los logs:

```
Ingesta finalizada. Procesados=17 | OK=17 | Fallidos=0
```

Los logs detallados se guardan en `logs/rag_pipeline.log`. Si algún documento falló, el nombre del archivo y el error aparecen en ese archivo.

### Reingestión incremental

Si ejecutás la ingesta nuevamente sobre documentos ya procesados, el sistema los detecta y los saltea automáticamente. Solo procesa los archivos nuevos.

## 5. Interfaz de consulta

### Levantar la aplicación

Asegurate de que Ollama está corriendo (`ollama serve`) y luego:

```bash
make app
```

Abrir en el navegador: <http://localhost:8501>

### Hacer una consulta

1. Escribí tu pregunta en el campo de texto inferior.
2. Presioná **Enter** o el botón de envío.
3. El sistema busca en los documentos indexados y genera una respuesta.

**Ejemplos de consultas útiles:**

- *¿Qué problemas de circulación se registraron en el pozo PM-104?*
- *¿Cuáles son las recomendaciones para pozos con pérdida de circulación en Quintuco?*
- *¿Qué causó la falla del BES en el pozo LL-112?*
- *¿Qué presión de inyección se usó en la recuperación secundaria del bloque sur?*

### Entender las respuestas

El sistema genera respuestas basadas exclusivamente en los documentos indexados. Si no encuentra información relevante, responde indicando que no encontró datos suficientes — no inventa información.

El historial de la conversación se mantiene durante la sesión, por lo que podés hacer preguntas de seguimiento sobre el mismo tema.

### Cerrar la aplicación

Presioná `Ctrl + C` en la terminal donde corre `make app`.

## 6. Evaluación de modelos

El sistema incluye dos scripts de evaluación opcionales para medir la calidad del pipeline.

### Evaluación de embeddings

Mide la calidad del modelo de embeddings para recuperar documentos relevantes:

```bash
make evaluate
```

El resultado se guarda en `evaluation_results.json`.

### Evaluación comparativa de LLMs

Compara el rendimiento de distintos modelos de lenguaje sobre consultas reales del dominio:

```bash
make evaluate-llm
```

Por defecto compara `llama3.1:8b`, `mistral:7b`, `qwen3:8b`, `llama3.2:3b` y `gemma2:9b`. Para personalizar:

```bash
make evaluate-llm ARGS="--models llama3.1:8b qwen3:8b --judge-model qwen3:8b --n-runs 3 --output eval_results/mi_reporte.json"
```

Parámetros disponibles:

| Parámetro | Descripción |
| --- | --- |
| `--models` | Modelos a comparar (separados por espacio) |
| `--judge-model` | Modelo juez para medir corrección y fidelidad |
| `--n-runs` | Repeticiones por consulta (1 para pruebas rápidas) |
| `--min-faithfulness` | Fidelidad mínima exigida (escala 1–5) |
| `--latency-penalty` | Penalización por latencia alta |
| `--output` | Ruta del reporte JSON |

El reporte se guarda en `eval_results/report.json` por defecto.

## 7. Solución de problemas frecuentes

### "Ollama no está corriendo"

La app muestra este error si no puede conectarse al servidor de Ollama.

**Solución:** Abrí una terminal y ejecutá:

```bash
ollama serve
```

Si Ollama corre en una máquina diferente, actualizá la variable `OLLAMA_BASE_URL` en `.env`.

### La app carga pero no responde preguntas

Puede que el vectorstore esté vacío — nunca se corrió la ingesta o se borró `data/vectorstore/`.

**Solución:** Ejecutar la ingesta:

```bash
make run
```

### Error de permisos con Docker

En Linux, Docker puede requerir `sudo`.

**Solución (permanente):** Agregá tu usuario al grupo docker:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### `make build` falla con error en `/cache`

En algunas máquinas el directorio `/cache/huggingface` no se crea automáticamente.

**Solución:** En el `Dockerfile`, agregá esta línea antes del `RUN chmod -R 777 /cache`:

```dockerfile
RUN mkdir -p /cache/huggingface
```

### La ingesta es muy lenta

El procesamiento de PDFs con Docling es intensivo. Los tiempos normales son:

- PDFs simples (1–5 páginas): 2–10 segundos
- PDFs complejos con tablas o imágenes: 5–15 minutos

Podés aumentar el paralelismo si tenés más núcleos disponibles:

```bash
docker run --rm -v ./data:/app/data rag-app python ingest_files.py --max-workers 4
```
