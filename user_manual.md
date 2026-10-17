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
└── sedimentologia/    # Estudios sedimentológicos
```

El tipo de carpeta determina cómo se clasifica el documento en el sistema.

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
