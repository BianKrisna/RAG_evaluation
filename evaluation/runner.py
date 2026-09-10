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


def _make_target(rag_chain) -> Callable:
    """
    Wrap a RAG chain into the target function signature that LangSmith expects:
        target(inputs: dict) -> dict

    Chains that return {"answer": ..., "context": ...} work directly.
    Chains that return a plain string are wrapped automatically.

    The output dict MUST contain:
        "answer"  (str) — the generated answer
        "context" (str) — retrieved docs joined as text (for groundedness /
                          retrieval_relevance metrics). If your chain doesn't
                          expose context, pass context_key=None and those two
                          metrics will receive an empty string.
    """

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        raw = rag_chain.invoke({"input": question})

        # Handle dict output (create_retrieval_chain style)
        if isinstance(raw, dict):
            answer = raw.get("answer", raw.get("output", ""))
            # Flatten retrieved docs to a single string
            source_docs = raw.get("context", [])
            if isinstance(source_docs, list):
                context = "\n\n".join(
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                    for doc in source_docs
                )
            else:
                context = str(source_docs)
        # Handle plain-string output (LCEL pipe chains)
        else:
            answer = str(raw)
            context = ""

        return {"answer": answer, "context": context}

    return target


def evaluate_rag(
    rag_chain,
    dataset_name: str,
    experiment_prefix: str = "rag-eval",
    evaluators: list[Callable] | None = None,
    client: Client | None = None,
    max_concurrency: int = 1,
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
        **kwargs:          Extra args forwarded to langsmith.evaluate().

    Returns:
        langsmith EvaluationResults object.
    """
    from evaluation.evaluators import ALL_EVALUATORS

    evaluators = evaluators if evaluators is not None else ALL_EVALUATORS
    client = client or Client()

    results = evaluate(
        _make_target(rag_chain),
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        client=client,
        max_concurrency=max_concurrency,
        **kwargs,
    )
    return results
