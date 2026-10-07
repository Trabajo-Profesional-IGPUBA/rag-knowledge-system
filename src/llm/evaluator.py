"""Evaluación comparativa de modelos LLM sobre el pipeline RAG (latencia, cobertura de keywords)."""

import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.embedder import Embedder
from src.llm.client import LLMClient, LLMConfig
from src.llm.eval_queries import EVAL_QUERIES, NO_INFO_PATTERNS
from src.llm.prompt_builder import PromptBuilder
from src.llm.rag_pipeline import RAGConfig, RAGPipeline
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

# Pesos de cada componente en la calidad de una respuesta (se renormalizan
# si algún componente no está disponible, p. ej. sin judge).
QUALITY_WEIGHTS = {"judge": 0.40, "keywords": 0.25, "numbers": 0.25, "semantic": 0.10}


@dataclass
class ModelEvalResult:
    """Resultado de evaluar un modelo sobre una consulta puntual."""

    model: str
    query_id: str
    query: str
    response: str
    elapsed_sec: float
    response_length: int
    keyword_hits: int
    keyword_total: int
    keyword_score: float
    run: int = 1
    error: str | None = None
    is_no_info_response: bool = False
    semantic_similarity: float | None = None
    quality: float = 0.0
    number_score: float | None = None
    judge_correctness: float | None = None
    judge_completeness: float | None = None  # 1-5
    judge_faithfulness: float | None = None  # 1-5
    should_abstain: bool = False
    abstained: bool = False
    context_recall: float | None = None
    context_ok: bool | None = None
    context: str = ""

    @property
    def ok(self) -> bool:
        """True si la consulta se ejecutó sin error."""
        return self.error is None


@dataclass
class EvaluationReport:
    """Reporte consolidado de la evaluación: resultados por query, resumen por modelo y selección final."""

    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    models_evaluated: list[str] = field(default_factory=list)
    results: list[ModelEvalResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    selected_model: str = ""
    selection_rationale: str = ""

    def to_dict(self) -> dict:
        """Convierte el reporte a diccionario serializable."""
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        """Guarda el reporte como JSON en el path indicado, creando carpetas si hace falta."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Reporte de evaluación guardado en %s", path)

    def print_summary(self) -> None:
        """Imprime en consola un resumen legible del reporte por modelo."""
        print("\n" + "═" * 60)
        print("EVALUACIÓN COMPARATIVA DE MODELOS LLM")
        print("═" * 60)
        for model, stats in self.summary.items():
            halluc = (
                f"{stats['hallucination_rate']:.0%}"
                if stats["total_negative"]
                else "n/d"
            )
            print(f"\n  Modelo: {model}")
            print(f"    Calidad global (errores=0) : {stats['quality']:.0%}")
            print(f"    Calidad (contexto con resp.): {stats['quality_ctx_ok']:.0%}")
            print(
                f"    Keywords / Números         : {stats['avg_keyword_score']:.0%} / {stats['avg_number_score']:.0%}"
            )
            print(
                f"    Judge correct. / fidelidad : {stats['avg_judge_correctness']:.1f} / {stats['avg_judge_faithfulness']:.1f} (de 5)"
            )
            print(f"    Latencia promedio : {stats['avg_elapsed_sec']:.2f}s")
            print(f"    Resp. promedio    : {stats['avg_response_length']:.0f} chars")
            print(
                f"    Abstención incorrecta      : {stats['false_abstention_rate']:.0%}"
            )
            print(f"    Alucinación (sin respuesta): {halluc}")
            print(f"    Recall del retriever       : {stats['retrieval_recall']:.0%}")
            print(
                f"    Errores                    : {stats['error_count']:.0f} de {stats['total_runs']:.0f}"
            )
        print(f"\n  → Modelo seleccionado: {self.selected_model}")
        print(f"  → Justificación: {self.selection_rationale}")
        print("═" * 60)


def _normalize(text: str) -> str:
    """Normaliza texto: minúsculas y sin acentos, para matching más tolerante."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text


def _as_groups(keywords: list) -> list[list[str]]:
    """Convierte ['a', ['b', 'c']] en [['a'], ['b', 'c']] (grupos de sinónimos)."""
    return [[k] if isinstance(k, str) else list(k) for k in keywords]


def _score_keywords(
    text: str, keywords: list, query: str = ""
) -> tuple[int, int, float]:
    q_norm = _normalize(query)
    t_norm = _normalize(text)
    groups = [
        g for g in _as_groups(keywords) if not any(_normalize(a) in q_norm for a in g)
    ]
    hits = sum(1 for g in groups if any(_normalize(a) in t_norm for a in g))
    total = len(groups)
    return hits, total, (hits / total if total else 0.0)


def _nums(text: str) -> set[str]:
    """Números sin separadores ('2.450' == '2450'); ignora los de 1 dígito."""
    found = re.findall(r"\d+(?:[.,]\d+)*", text)
    cleaned = {n.replace(".", "").replace(",", "") for n in found}
    return {n for n in cleaned if len(n) >= 2}


def _score_numbers(text: str, reference: str | None, query: str = "") -> float | None:
    """Fracción de los números de la referencia que aparecen en `text`."""
    if not reference:
        return None
    ref = _nums(reference) - _nums(query)
    if not ref:
        return None
    return len(ref & _nums(text)) / len(ref)


def _run_quality(r: ModelEvalResult) -> float:
    """Calidad 0-1 de una corrida. Error = 0. Negativas: 1 si se abstuvo, 0 si no."""
    if not r.ok:
        return 0.0
    if r.should_abstain:
        return 1.0 if r.abstained else 0.0

    components: dict[str, float] = {}
    if r.judge_correctness is not None:
        components["judge"] = (r.judge_correctness - 1) / 4
    if r.keyword_total > 0:
        components["keywords"] = r.keyword_score
    if r.number_score is not None:
        components["numbers"] = r.number_score
    if r.semantic_similarity is not None:
        # reescala: 0.5 o menos → 0, 1.0 → 1 (el coseno crudo casi no separa)
        components["semantic"] = max(0.0, (r.semantic_similarity - 0.5) / 0.5)

    if not components:
        return 0.0
    weight_sum = sum(QUALITY_WEIGHTS[k] for k in components)
    return sum(QUALITY_WEIGHTS[k] * v for k, v in components.items()) / weight_sum


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _summarize(results: list[ModelEvalResult]) -> dict[str, float]:
    answerable = [r for r in results if not r.should_abstain]
    negatives = [r for r in results if r.should_abstain]
    ok = [r for r in results if r.ok]
    ok_ans = [r for r in answerable if r.ok]
    ok_neg = [r for r in negatives if r.ok]

    return {
        "quality": _mean([r.quality for r in results]),  # errores cuentan como 0
        "quality_ctx_ok": _mean(
            [r.quality for r in answerable if r.context_ok is not False]
        ),
        "avg_elapsed_sec": _mean([r.elapsed_sec for r in ok]),
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
        "avg_response_length": _mean([float(r.response_length) for r in ok]),
        "no_info_rate": _mean([1.0 if r.is_no_info_response else 0.0 for r in ok]),
        "error_count": float(len(results) - len(ok)),
        "ok_runs": float(len(ok)),
        "total_runs": float(len(results)),
        "total_negative": float(len(negatives)),
    }


def _is_no_info_response(response: str) -> bool:
    """Detecta si la respuesta es una abstención ('no encontré información...')."""
    response_norm = _normalize(response)
    return any(_normalize(p) in response_norm for p in NO_INFO_PATTERNS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _semantic_similarity(
    response: str, reference: str | None, embedder: Embedder | None
) -> float | None:
    """Similitud semántica (coseno) entre la respuesta generada y la referencia."""
    if not reference or embedder is None:
        return None
    resp_emb = embedder.embed(response)
    ref_emb = embedder.embed(reference)
    return round(_cosine_similarity(resp_emb, ref_emb), 4)


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


def _select_best_model(summary: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Selecciona el mejor modelo por calidad combinada (keywords + similitud
    semántica), desempatando por menor latencia. Devuelve (modelo, justificación)."""
    if not summary:
        return "", ""

    def _quality(stats: dict[str, float]) -> float:
        return (stats["avg_keyword_score"] + stats["avg_semantic_similarity"]) / 2

    best_model, stats = max(
        summary.items(),
        key=lambda x: (_quality(x[1]), -x[1]["avg_elapsed_sec"]),
    )
    rationale = (
        f"Mayor calidad combinada (keywords {stats['avg_keyword_score']:.0%} "
        f"+ similitud semántica {stats['avg_semantic_similarity']:.0%}) "
        f"con latencia de {stats['avg_elapsed_sec']:.1f}s promedio. "
        f"Tasa de abstención: {stats['no_info_rate']:.0%}."
    )
    return best_model, rationale


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

        for q in queries:
            for run in range(1, n_runs + 1):
                res = _evaluate_one(pipeline, q, model_name, run, embedder)
                if judge_client is not None and res.ok:
                    _apply_judge(res, q, judge_client)
                report.results.append(res)

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
