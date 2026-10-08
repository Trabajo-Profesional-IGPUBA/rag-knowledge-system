"""Evaluación comparativa de modelos LLM sobre el pipeline RAG (latencia, cobertura de keywords)."""

import logging
import time
from pathlib import Path
from typing import Any

from src.embeddings.embedder import Embedder
from src.llm.client import LLMClient, LLMConfig
from src.llm.eval_context import (
    CONTEXT_OK_THRESHOLD,
    _extract_context,
)
from src.llm.eval_judge import _judge_all
from src.llm.eval_models import EvaluationReport, ModelEvalResult
from src.llm.eval_quality import _run_quality
from src.llm.eval_queries import EVAL_QUERIES
from src.llm.eval_selection import LATENCY_PENALTY_PER_DOUBLING, _select_best_model
from src.llm.eval_summary import _summarize
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


def evaluate_models(
    retriever: Retriever,
    models: list[str],
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    embedder: Embedder | None = None,
    judge_model: str | None = None,
    n_runs: int = 3,
    latency_penalty: float = LATENCY_PENALTY_PER_DOUBLING,
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
            report.summary,
            latency_penalty=latency_penalty,
        )

    if output_path:
        report.save(output_path)

    report.print_summary()
    return report
