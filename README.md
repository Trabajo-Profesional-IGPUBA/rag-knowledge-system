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
