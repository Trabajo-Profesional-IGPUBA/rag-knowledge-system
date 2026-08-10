# RAG Knowledge System

Sistema de Generación Aumentada por Recuperación (RAG) para la gestión de conocimiento técnico en cuencas maduras.

## Descripción

Este sistema permite consultar inteligentemente documentación histórica de pozos petroleros (informes de perforación, partes diarios, análisis de falla, estudios de reservorio, entre otros) mediante lenguaje natural.

## Estructura del proyecto

```
rag-knowledge-system/
├── data/
│   ├── raw/            # Documentos originales
│   ├── processed/      # Texto procesado y segmentado
│   └── vectorstore/    # Base de datos vectorial
├── src/
│   ├── etl/            # Ingesta, OCR y chunking
│   ├── embeddings/     # Generación de vectores
│   ├── retrieval/      # Búsqueda semántica
│   ├── llm/            # Integración del modelo de lenguaje
│   └── interface/      # API o interfaz de consulta
└── tests/              # Pruebas unitarias y de integración
```

## Instalación

```bash
git clone https://github.com/tu-usuario/rag-knowledge-system.git
cd rag-knowledge-system
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
## Docker

### Levantar la aplicación

```bash
docker compose up app
```

Inicia el contenedor de la aplicación y sus dependencias.

### Ejecutar los tests

```bash
docker compose run test
```

Ejecuta las pruebas del proyecto dentro de un contenedor.

### Formatear el código

```bash
docker compose run format
```

Aplica automáticamente las reglas de formato configuradas en el proyecto.

### Verificar el estilo del código (lint)

```bash
docker compose run lint
```

Analiza el código para detectar errores de estilo, problemas de calidad o incumplimientos de las reglas definidas por el linter y las corrige.

### Ejecucion del indexer 
```bash
python -m src.indexer
```

### Ejecucion de la interfaz web
```bash
streamlit run app.py --server.fileWatcherType none 
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
