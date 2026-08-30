
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

### Evaluación de modelos de embeddings
```bash
docker compose run evaluate_embeddings
```

## Lint y formato de código

El proyecto usa `ruff` (lint) y `black` (formato), ambos ya incluidos en
la imagen Docker. El CI verifica automáticamente que el código cumpla
ambos estándares y bloquea el PR si no es así.

### Verificar localmente antes de commitear

```bash
docker compose run lint-check     # solo reporta errores de lint
docker compose run format-check   # solo reporta problemas de formato
```

### Corregir automáticamente

```bash
docker compose run lint     # aplica fixes de ruff
docker compose run format   # aplica formato de black
```

### Hook de pre-commit (recomendado)

Instalar dependencias de desarrollo (una sola vez):

```bash
pip install -r requirements-dev.txt
pre-commit install
```

A partir de ahí, cada `git commit` va a correr lint y formato
automáticamente sobre los archivos modificados, usando los mismos
contenedores Docker del proyecto.

Para correrlo manualmente sobre todo el repo sin hacer commit:

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