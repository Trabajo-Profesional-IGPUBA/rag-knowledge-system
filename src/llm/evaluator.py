"""Evaluación comparativa de modelos LLM sobre el pipeline RAG (latencia, cobertura de keywords)."""

import json
import logging
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from src.embeddings.embedder import Embedder
from src.llm.client import LLMClient, LLMConfig
from src.llm.eval_models import EvaluationReport, ModelEvalResult
from src.llm.eval_quality import _run_quality
from src.llm.eval_queries import EVAL_QUERIES
from src.llm.eval_text import (
    _is_no_info_response,
    _score_keywords,
    _score_numbers,
    _semantic_similarity,
)
from src.llm.prompt_builder import PromptBuilder
from src.llm.rag_pipeline import RAGConfig, RAGPipeline
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

LATENCY_PENALTY_PER_DOUBLING = 0.05


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Percentil por nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1))
    return round(ordered[idx], 2)


def _summarize(results: list[ModelEvalResult]) -> dict[str, float]:
    answerable = [r for r in results if not r.should_abstain]
    negatives = [r for r in results if r.should_abstain]
    ok = [r for r in results if r.ok]
    ok_ans = [r for r in answerable if r.ok]
    ok_neg = [r for r in negatives if r.ok]
    latencies = [r.elapsed_sec for r in ok]

    return {
        "quality": _mean([r.quality for r in results]),  # errores cuentan como 0
        "quality_ctx_ok": _mean(
            [r.quality for r in answerable if r.context_ok is not False]
        ),
        "avg_keyword_score": _mean([r.keyword_score for r in ok]),
        "avg_number_score": _mean(
            [r.number_score for r in ok if r.number_score is not None]
        ),
        "avg_semantic_similarity": _mean(
            [r.semantic_similarity for r in ok if r.semantic_similarity is not None]
        ),
        "avg_judge_correctness": _mean(
            [r.judge_correctness for r in ok if r.judge_correctness is not None]
        ),
        "avg_judge_faithfulness": _mean(
            [r.judge_faithfulness for r in ok if r.judge_faithfulness is not None]
        ),
        "retrieval_recall": _mean(
            [r.context_recall for r in answerable if r.context_recall is not None]
        ),
        "false_abstention_rate": _mean([1.0 if r.abstained else 0.0 for r in ok_ans]),
        "hallucination_rate": _mean([0.0 if r.abstained else 1.0 for r in ok_neg]),
        "avg_elapsed_sec": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "p50_elapsed_sec": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_elapsed_sec": _percentile(latencies, 0.95),
        "avg_response_length": _mean([float(r.response_length) for r in ok]),
        "no_info_rate": _mean([1.0 if r.is_no_info_response else 0.0 for r in ok]),
        "error_count": float(len(results) - len(ok)),
        "ok_runs": float(len(ok)),
        "total_runs": float(len(results)),
        "total_negative": float(len(negatives)),
    }


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
        f"Contexto recuperado:\n{context[:4000] or '(vacío)'}\n\n"
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
        log.warning("Judge falló en '%s': %s", query[:40], e)
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
    r.quality = round(_run_quality(r), 4)


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


def _select_best_model(
    summary: dict[str, dict[str, float]],
    latency_penalty: float = LATENCY_PENALTY_PER_DOUBLING,
) -> tuple[str, str]:
    """Elige el modelo que mejor responde sin aceptar una latencia desproporcionada."""
    if not summary:
        return "", ""

    def _latency(m: str) -> float:
        return max(summary[m]["p95_elapsed_sec"], 0.01)

    fastest = min(_latency(m) for m in summary)

    def _adjusted(m: str) -> float:
        return summary[m]["quality"] - latency_penalty * math.log2(
            _latency(m) / fastest
        )

    best = max(
        summary,
        key=lambda m: (
            round(_adjusted(m), 6),
            -_latency(m),
            -summary[m]["p50_elapsed_sec"],
        ),
    )

    s = summary[best]
    rationale = (
        f"Calidad {s['quality']:.0%} con latencia máxima (p95) de "
        f"{s['p95_elapsed_sec']:.1f}s (típica p50 {s['p50_elapsed_sec']:.1f}s): "
        "es el mejor equilibrio entre calidad y tiempo de respuesta."
    )
    return best, rationale


def _evaluate_one(pipeline, q, model_name, run, embedder) -> ModelEvalResult:
    """Ejecuta una query una vez y calcula las métricas."""
    keywords = q.get("expected_keywords", [])
    reference = q.get("reference_answer")
    should_abstain = bool(q.get("should_abstain", False))
    query = q["query"]

    try:
        t0 = time.perf_counter()
        rag_resp = pipeline.query(query)
        elapsed = time.perf_counter() - t0
        answer = rag_resp.answer
        context = _extract_context(rag_resp)

        hits, total, kw_score = _score_keywords(answer, keywords, query)
        num_score = _score_numbers(answer, reference, query)
        sem_sim = _semantic_similarity(answer, reference, embedder)

        # Contexto: ¿tenía los hechos esperados? (separa falla del retriever vs del LLM)
        recall_parts = []
        if context and total > 0:
            recall_parts.append(_score_keywords(context, keywords, query)[2])
        if context:
            ctx_num = _score_numbers(context, reference, query)
            if ctx_num is not None:
                recall_parts.append(ctx_num)
        ctx_recall = sum(recall_parts) / len(recall_parts) if recall_parts else None
        ctx_ok = None if ctx_recall is None else ctx_recall >= CONTEXT_OK_THRESHOLD

        pattern_abstain = _is_no_info_response(answer)
        result = ModelEvalResult(
            model=model_name,
            query_id=q["id"],
            query=query,
            response=answer,
            elapsed_sec=round(elapsed, 2),
            response_length=len(answer),
            keyword_hits=hits,
            keyword_total=total,
            keyword_score=round(kw_score, 4),
            run=run,
            should_abstain=should_abstain,
            abstained=pattern_abstain,
            is_no_info_response=pattern_abstain,
            semantic_similarity=sem_sim,
            number_score=None if num_score is None else round(num_score, 4),
            context_recall=None if ctx_recall is None else round(ctx_recall, 4),
            context_ok=ctx_ok,
            context=context[:4000],
        )
    except Exception as e:
        log.error("Error evaluando %s en %s: %s", model_name, q["id"], e)
        result = ModelEvalResult(
            model=model_name,
            query_id=q["id"],
            query=query,
            response="",
            elapsed_sec=0.0,
            response_length=0,
            keyword_hits=0,
            keyword_total=0,
            keyword_score=0.0,
            run=run,
            error=str(e),
            should_abstain=should_abstain,
            number_score=0.0 if reference else None,
            semantic_similarity=0.0 if reference else None,
        )

    result.quality = round(_run_quality(result), 4)
    return result


def _model_available(model: str, available: list[str]) -> bool:
    """Sin tag, Ollama lo resuelve como ':latest'."""
    names = set(available)
    return model in names or f"{model}:latest" in names


def _ensure_model(client: LLMClient, model: str) -> bool:
    """Verifica Ollama y descarga el modelo si falta. False si no se puede usar."""
    if not client.is_available():
        log.error("Ollama no disponible para modelo %s", model)
        return False
    if not _model_available(model, client.list_models()):
        log.info("Modelo %s no encontrado, descargando...", model)
        client.pull_model(model)
    return True


CONTEXT_OK_THRESHOLD = 0.5
_CHUNK_TEXT_KEYS = ("text", "content", "document", "chunk_text", "page_content")


def _extract_context(rag_resp: Any) -> str:
    """Texto de los chunks que realmente entraron al prompt del LLM."""
    chunks = rag_resp.retrieval.chunks[: rag_resp.prompt.num_chunks]
    parts = []
    for c in chunks:
        text = next((c[k] for k in _CHUNK_TEXT_KEYS if c.get(k)), None)
        if text:
            parts.append(str(text))
    if not parts:
        raise RuntimeError(
            f"No encontré el texto del chunk. Claves disponibles: {list(chunks[0].keys()) if chunks else 'sin chunks'}"
        )
    return "\n---\n".join(parts)


def evaluate_models(
    retriever: Retriever,
    models: list[str],
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    embedder: Embedder | None = None,
    judge_model: str | None = None,
    n_runs: int = 3,
) -> EvaluationReport:
    """Evalúa cada modelo contra el set de queries, arma el resumen y selecciona el mejor."""
    queries = queries or EVAL_QUERIES
    report = EvaluationReport(models_evaluated=models)
    prompt_builder = PromptBuilder()
    judge_client: LLMClient | None = None
    if judge_model:
        if judge_model in models:
            log.warning(
                "El judge (%s) está entre los evaluados: se juzgaría a sí mismo.",
                judge_model,
            )
        candidate = LLMClient(
            LLMConfig(
                model=judge_model,
                temperature=0.0,
                think=False if judge_model.startswith("qwen3") else None,
            )
        )
        if _ensure_model(candidate, judge_model):
            judge_client = candidate
    else:
        log.info("Sin judge_model: se omite LLM-as-judge (abstención por patrones).")

    for model_name in models:
        log.info("Evaluando modelo: %s", model_name)
        print(f"\n[Evaluando {model_name}...]")

        config = LLMConfig(model=model_name, temperature=0.1)
        client = LLMClient(config)

        if not _ensure_model(client, model_name):
            continue

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=client,
            prompt_builder=prompt_builder,
            config=RAGConfig(top_k=5, min_score=0.2),
        )

        # Warm-up: carga el modelo en memoria para que no contamine la latencia medida
        try:
            log.info("Warm-up de %s...", model_name)
            pipeline.query(queries[0]["query"])
        except Exception as e:
            log.warning("Warm-up falló para %s: %s", model_name, e)

        for q in queries:
            for run in range(1, n_runs + 1):
                report.results.append(
                    _evaluate_one(pipeline, q, model_name, run, embedder)
                )

    # Fase 2: el juez corre al final, una sola vez, para no mezclar su carga con los
    # tiempos de los modelos evaluados ni forzar recargas de modelos en cada consulta.
    if judge_client is not None and report.results:
        _judge_all(report.results, queries, judge_client)

    for model_name in models:
        model_results = [r for r in report.results if r.model == model_name]
        if model_results:
            report.summary[model_name] = _summarize(model_results)

    if report.summary:
        report.selected_model, report.selection_rationale = _select_best_model(
            report.summary
        )

    if output_path:
        report.save(output_path)

    report.print_summary()
    return report
