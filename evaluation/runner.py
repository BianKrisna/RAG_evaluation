"""
runner.py — Generic LangSmith evaluate() wrapper for any RAG chain.

Usage:
    from evaluation import evaluate_rag, ALL_EVALUATORS

    results = evaluate_rag(
        rag_chain=my_chain,          # any LangChain Runnable
        dataset_name="my-dataset",
        experiment_prefix="my-rag",
        evaluators=ALL_EVALUATORS,   # or pick a subset
    )
"""

from __future__ import annotations
from typing import Any, Callable

from langsmith import evaluate, Client
from langchain_core.documents import Document
from rag.evidence import evidence_record, format_evidence


def summarize_evaluations(results) -> dict[str, dict[str, int]]:
    """Count scored/failed examples per metric from the explicit error flags.

    Counts describe evaluation coverage, not answer accuracy. A valid zero grade
    is scored successfully; a null quality score is an evaluation error.
    """
    from .evaluators import ERROR_SUFFIX

    summary = {}
    for row in results:
        for feedback in row["evaluation_results"]["results"]:
            if feedback.key.endswith(ERROR_SUFFIX):
                metric = feedback.key[:-len(ERROR_SUFFIX)]
                counts = summary.setdefault(metric, {"scored": 0, "errors": 0})
                counts["errors" if feedback.score == 1 else "scored"] += 1
    return summary


def _make_target(rag_chain) -> Callable:
    """
    Wrap a RAG chain into the target function signature that LangSmith expects:
        target(inputs: dict) -> dict

    Chains that return {"answer": ..., "context": ...} work directly.
    Chains that return a plain string are wrapped automatically.

    The output dict MUST contain:
        "answer"  (str) — the generated answer
        "context" (str) — retrieved docs with source/ID/page headers (for the judges).
        "source_documents" retains structured evidence, and citation fields are
        forwarded when present. Chains without context receive an empty string.
    """

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        raw = rag_chain.invoke({"input": question})
        source_records = []
        citation_fields = {}

        # Handle dict output (create_retrieval_chain style)
        if isinstance(raw, dict):
            answer = raw.get("answer", raw.get("output", ""))
            # Flatten retrieved docs to a single string
            source_docs = raw.get("context", [])
            if isinstance(source_docs, list):
                context = "\n\n".join(
                    format_evidence(doc) if isinstance(doc, Document) else str(doc)
                    for doc in source_docs
                )
                source_records = [evidence_record(doc) for doc in source_docs
                                  if isinstance(doc, Document)]
            else:
                context = str(source_docs)
            citation_fields = {key: raw[key] for key in ("citations", "citation_validation")
                               if key in raw}
        # Handle plain-string output (LCEL pipe chains)
        else:
            answer = str(raw)
            context = ""

        return {"answer": answer, "context": context,
                "source_documents": source_records, **citation_fields}

    return target


def evaluate_rag(
    rag_chain,
    dataset_name: str,
    experiment_prefix: str = "rag-eval",
    evaluators: list[Callable] | None = None,
    client: Client | None = None,
    max_concurrency: int = 1,
    data=None,
    **kwargs: Any,
):
    """
    Run LangSmith evaluation on any RAG chain.

    Args:
        rag_chain:         LangChain Runnable. Must accept {"input": question}.
        dataset_name:      LangSmith dataset name (create with create_or_get_dataset).
        experiment_prefix: Label prefix shown in LangSmith UI.
        evaluators:        List of evaluator functions. Defaults to ALL_EVALUATORS.
        client:            LangSmith Client (auto-created from env if None).
        max_concurrency:   Parallel eval calls.
        data:              Optional shared dataset snapshot; otherwise uses dataset_name.
        **kwargs:          Extra args forwarded to langsmith.evaluate().

    Returns:
        langsmith EvaluationResults object.
    """
    from evaluation.evaluators import ALL_EVALUATORS

    evaluators = evaluators if evaluators is not None else ALL_EVALUATORS
    client = client or Client()

    results = evaluate(
        _make_target(rag_chain),
        data=dataset_name if data is None else data,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        client=client,
        max_concurrency=max_concurrency,
        **kwargs,
    )
    return results
