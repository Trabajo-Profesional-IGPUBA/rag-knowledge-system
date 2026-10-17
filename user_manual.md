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