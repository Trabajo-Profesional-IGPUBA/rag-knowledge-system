# Manual de Usuario — Sistema RAG IGPUBA

## 1. Introducción

El Sistema RAG IGPUBA permite consultar documentación técnica de pozos petroleros en lenguaje natural. El usuario escribe preguntas como *"¿Qué problemas de circulación hubo en el pozo PM-104?"* y el sistema busca en los documentos indexados y genera una respuesta basada en ellos.

El sistema tiene dos operaciones principales:

1. **Ingesta**: procesar PDFs técnicos y cargarlos en la base de datos vectorial.
2. **Consulta**: interactuar con el sistema a través de una interfaz de chat web.

Este manual está dirigido a técnicos o ingenieros que necesiten operar el sistema. No se requiere conocimiento de programación.