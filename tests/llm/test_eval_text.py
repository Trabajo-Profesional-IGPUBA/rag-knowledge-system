"""
Cubre "ÉPICA: Cliente LLM sobre Ollama":
  - Medición de calidad de respuesta -> CA-11.1 a CA-11.3

Cubre "Mejorar precisión del scoring en la evaluación de LLMs: el matching exacto de keywords
subestimaba la calidad real de las respuestas":
  - Normalización de keywords -> CA-16.1
  - Detección de abstención -> CA-17.1
  - Similitud semántica-> CA-18.1, 18.2, 18.4 y 18.5
  - Selección del modelo con criterio combinado -> CA-19.1 y 19.3

Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Palabras clave esperadas-> CA-20.1 a CA-20.2
  - Datos numéricos-> CA-21.1 a CA-21.3
"""

from unittest.mock import MagicMock


class Testeval_text:
    # CA-11.1: El sistema debe calcular qué proporción de palabras clave esperadas aparece en una respuesta generada.
    def test_keyword_scoring(self):
        from src.llm.eval_text import _score_keywords

        response = "La presión de fondo del pozo PM-104 es 3500 psi"
        keywords = ["presión", "psi", "fondo", "PM-104"]
        hits, total, score = _score_keywords(response, keywords)
        assert hits == 4
        assert total == 4
        assert score == 1.0

    # CA-11.2: El sistema debe calcular correctamente el score cuando solo una parte de las palabras clave está presente.
    def test_keyword_scoring_partial(self):
        from src.llm.eval_text import _score_keywords

        response = "La presión es alta"
        keywords = ["presión", "psi", "fondo"]
        hits, total, score = _score_keywords(response, keywords)
        assert hits == 1
        assert total == 3
        assert round(score, 4) == round(1 / 3, 4)

    # CA-11.3: Si no hay palabras clave esperadas para una consulta, el sistema no debe fallar y debe devolver un score de cero.
    def test_keyword_scoring_empty_keywords(self):
        from src.llm.eval_text import _score_keywords

        hits, total, score = _score_keywords("cualquier texto", [])
        assert hits == 0
        assert total == 0.0
        assert score == 0.0


class TestNormalization:
    # CA-16.1: El sistema debe normalizar acentos y mayúsculas al comparar palabras
    # clave esperadas contra la respuesta generada.
    def test_normalize_removes_accents_and_lowercases(self):
        from src.llm.eval_text import _normalize

        assert _normalize("Pérdida de Circulación") == "perdida de circulacion"

    # CA-16.1: El sistema debe normalizar acentos y mayúsculas al comparar palabras
    # clave esperadas contra la respuesta generada, para que variaciones ortográficas
    # no reduzcan el score injustamente.
    def test_score_keywords_matches_despite_accent_mismatch(self):
        from src.llm.eval_text import _score_keywords

        response = "Hubo perdida de circulacion en el pozo"
        keywords = ["pérdida de circulación"]
        hits, total, score = _score_keywords(response, keywords)
        assert hits == 1
        assert total == 1
        assert score == 1.0


class TestNoInfoDetection:
    # CA-17.1: El sistema debe distinguir cuándo una respuesta corresponde a una abstención explícita del modelo
    # (ej. "no encontré información"), en lugar de tratarla igual que una respuesta con contenido incorrecto.

    def test_detects_no_info_response(self):
        from src.llm.eval_text import _is_no_info_response

        assert _is_no_info_response(
            "No encontré información sobre esto en los documentos disponibles."
        )

    # CA-17.1: El sistema debe distinguir cuándo una respuesta corresponde a una abstención explícita del modelo
    # (ej. "no encontré información"), en lugar de tratarla igual que una respuesta con contenido incorrecto.
    def test_does_not_flag_normal_response(self):
        from src.llm.eval_text import _is_no_info_response

        assert not _is_no_info_response(
            "El pozo PM-104 tuvo pérdida de circulación en Quintuco."
        )


class TestSemanticSimilarity:
    # CA-18.2: Si no hay respuesta de referencia, el cálculo de similitud semántica
    # debe devolver None sin fallar.
    def test_returns_none_without_reference(self):
        from src.llm.eval_text import _semantic_similarity

        result = _semantic_similarity("cualquier respuesta", None, MagicMock())
        assert result is None

    # CA-18.4: Si no se provee un embedder, el cálculo de similitud semántica debe
    # devolver None sin fallar.
    def test_returns_none_without_embedder(self):
        from src.llm.eval_text import _semantic_similarity

        result = _semantic_similarity("respuesta", "referencia", None)
        assert result is None

    # CA-18.1: El sistema debe poder calcular la similitud semántica entre la
    # respuesta generada y una respuesta de referencia, cuando esta última esté
    # definida para la consulta.
    def test_identical_texts_have_similarity_close_to_one(self):
        from src.llm.eval_text import _semantic_similarity

        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]

        result = _semantic_similarity("mismo texto", "mismo texto", embedder)
        assert result == 1.0

    # CA-18.5: El cálculo de similitud semántica (coseno) no debe fallar ante un
    # vector nulo, debe devolver 0.0 en ese caso.
    def test_cosine_similarity_zero_vector_returns_zero(self):
        from src.llm.eval_text import _cosine_similarity

        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestKeywordGroups:
    # CA-20.1: El sistema debe aceptar varias formas equivalentes de decir lo mismo
    # para una palabra clave esperada (ej. "Water Cut" o "corte de agua"),
    # y contarla como un solo acierto si aparece cualquiera de ellas.

    def test_synonym_group_counts_as_single_hit(self):
        from src.llm.eval_text import _score_keywords

        hits, total, _ = _score_keywords(
            "el corte de agua subió", [["Water Cut", "corte de agua"], "CO2"]
        )
        assert hits == 1
        assert total == 2

    # CA-20.1: El sistema debe aceptar varias formas equivalentes de decir lo mismo
    # para una palabra clave esperada (ej. "Water Cut" o "corte de agua"),
    # y contarla como un solo acierto si aparece cualquiera de ellas.
    def test_as_groups_accepts_strings_and_lists(self):
        from src.llm.eval_text import _as_groups

        assert _as_groups(["a", ["b", "c"]]) == [["a"], ["b", "c"]]

    # CA-20.2: El sistema no debe sumar puntos por palabras clave
    # que ya están escritas en la propia pregunta.
    def test_ignores_keywords_present_in_query(self):
        from src.llm.eval_text import _score_keywords

        hits, total, score = _score_keywords(
            "Quintuco", ["Quintuco", "LCM"], query="¿Problemas en Quintuco?"
        )
        assert hits == 0
        assert total == 1
        assert score == 0.0


class TestNumberScoring:
    # CA-21.1: El sistema debe medir qué proporción de las cifras de la respuesta
    # de referencia aparece en la respuesta generada, sin que influya
    # cómo están escritos los separadores de miles (ej. "2.450" y "2450").
    def test_thousand_separators_are_equivalent(self):
        from src.llm.eval_text import _score_numbers

        assert _score_numbers("a 2450 metros", "a 2.450 metros") == 1.0

    # CA-21.1: El sistema debe medir qué proporción de las cifras de la respuesta
    # de referencia aparece en la respuesta generada, sin que influya
    # cómo están escritos los separadores de miles (ej. "2.450" y "2450").
    def test_partial_coverage(self):
        from src.llm.eval_text import _score_numbers

        assert _score_numbers("15", "15 m³/h y 2.450 m") == 0.5

    # CA-21.2: Si no hay respuesta de referencia, o ésta no contiene cifras,
    # la medición numérica debe devolver None sin fallar.
    def test_returns_none_without_reference(self):
        from src.llm.eval_text import _score_numbers

        assert _score_numbers("2450", None) is None

    # CA-21.2: Si no hay respuesta de referencia, o ésta no contiene cifras,
    # la medición numérica debe devolver None sin fallar.
    def test_returns_none_when_reference_has_no_numbers(self):
        from src.llm.eval_text import _score_numbers

        assert (
            _score_numbers("falló por fatiga", "falló por fatiga del material") is None
        )

    # CA-21.3: El sistema debe ignorar las cifras de un solo dígito
    # y las que ya aparecen en la pregunt
    def test_ignores_single_digit_numbers(self):
        from src.llm.eval_text import _score_numbers

        assert _score_numbers("etapa 4", "etapa 4") is None

    # CA-21.3: El sistema debe ignorar las cifras de un solo dígito
    # y las que ya aparecen en la pregunta.
    def test_ignores_numbers_present_in_query(self):
        from src.llm.eval_text import _score_numbers

        score = _score_numbers(
            "2450", "pozo PM-104 a 2.450", query="¿Qué pasó en PM-104?"
        )
        assert score == 1.0
