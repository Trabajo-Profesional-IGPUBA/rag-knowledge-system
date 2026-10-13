"""Evaluación con modelo juez."""

import json
import logging
import re
from typing import Any

from src.llm.client import LLMClient
from src.llm.eval_models import ModelEvalResult
from src.llm.eval_quality import _run_quality

log = logging.getLogger(__name__)

# Cantidad de decimales con que se redondea la calidad (0-1) de cada respuesta.
QUALITY_DECIMALS = 4

# Máximo de caracteres del contexto recuperado que se le pasa al juez.
JUDGE_MAX_CONTEXT_CHARS = 4000

# Máximo de caracteres de la consulta que se muestran en los logs.
LOG_QUERY_PREVIEW_CHARS = 40


def _generate(client: LLMClient, prompt: str) -> str:
    """Genera texto con un LLMClient (adaptar al método real del cliente)."""
    for name in ("generate", "complete", "chat", "ask"):
        fn = getattr(client, name, None)
        if callable(fn):
            out = fn(prompt)
            for attr in ("text", "content", "response", "answer"):
                if hasattr(out, attr):
                    return str(getattr(out, attr))
            return str(out)


def _judge(
    judge_client, query, response, context, reference, should_abstain
) -> dict | None:
    """Evalúa la respuesta con un modelo juez. Devuelve dict con puntajes o None si falla."""
    if should_abstain:
        ref_block = (
            "La pregunta NO tiene respuesta en los documentos. Lo correcto es que "
            "la respuesta se abstenga y diga que no hay información."
        )
    else:
        ref_block = f"Respuesta de referencia: {reference or '(no disponible)'}"

    prompt = (
        "Sos un evaluador estricto de un sistema RAG de ingeniería de petróleo.\n\n"
        f"Pregunta: {query}\n\n"
        f"{ref_block}\n\n"
        f"Contexto recuperado:\n{context[:JUDGE_MAX_CONTEXT_CHARS] or '(vacío)'}\n\n"
        f"Respuesta a evaluar:\n{response}\n\n"
        "Devolvé SOLO un JSON, sin texto extra, con esta forma exacta:\n"
        '{"correctness": 1-5, "completeness": 1-5, "faithfulness": 1-5, "abstained": true/false}\n'
        "- correctness: ¿coincide con la referencia (o con la abstención esperada)?\n"
        "- completeness: ¿cubre los hechos clave de la referencia?\n"
        "- faithfulness: 5 si TODO lo que afirma está en el contexto; 1 si inventa datos.\n"
        "- abstained: true si la respuesta dice que no encontró información."
    )
    try:
        raw = _generate(judge_client, prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except Exception as e:
        log.warning("Judge falló en '%s': %s", query[:LOG_QUERY_PREVIEW_CHARS], e)
        return None


def _apply_judge(
    r: ModelEvalResult, q: dict[str, Any], judge_client: LLMClient
) -> None:
    """Juzga una respuesta ya generada y actualiza sus puntajes, abstención y calidad."""
    verdict = _judge(
        judge_client,
        r.query,
        r.response,
        r.context,
        q.get("reference_answer"),
        r.should_abstain,
    )
    if not verdict:
        return

    def _f(key: str) -> float | None:
        try:
            return float(verdict[key]) if key in verdict else None
        except (TypeError, ValueError):
            return None

    r.judge_correctness = _f("correctness")
    r.judge_completeness = _f("completeness")
    r.judge_faithfulness = _f("faithfulness")
    if isinstance(verdict.get("abstained"), bool):
        r.abstained = verdict["abstained"]
    r.quality = round(_run_quality(r), QUALITY_DECIMALS)


def _judge_all(
    results: list[ModelEvalResult],
    queries: list[dict[str, Any]],
    judge_client: LLMClient,
) -> None:
    """Fase 2: el juez evalúa todas las respuestas juntas (se carga una sola vez)."""
    by_id = {q["id"]: q for q in queries}
    pending = [r for r in results if r.ok]
    log.info("Juzgando %d respuestas con el modelo juez...", len(pending))
    for i, r in enumerate(pending, start=1):
        log.info("  juez %d/%d: %s %s", i, len(pending), r.model, r.query_id)
        _apply_judge(r, by_id[r.query_id], judge_client)
