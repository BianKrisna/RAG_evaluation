"""
evaluation/evaluators.py

4 LLM-as-judge evaluators for RAG pipelines.

Scoring: 0-3 scale (normalized to 0.0-1.0 for LangSmith)
    0 = completely wrong / irrelevant / hallucinated
    1 = partially correct / somewhat relevant
    2 = mostly correct / relevant with minor issues
    3 = fully correct / highly relevant / well-grounded

Judge LLM: qwen/qwen3.8-27b via Groq.
Override model with JUDGE_MODEL env var. Uses GROQ_API_KEY.

Judge failures have no answer score and emit a separate <metric>_evaluation_error
flag. Each judge call gets at most three attempts for transient/validation errors.

Metrics:
    correctness         -- does answer match the reference? (needs ground truth)
    relevance           -- does answer actually address the question?
    groundedness        -- are all claims backed by retrieved context?
    retrieval_relevance -- are retrieved chunks on-topic for the question?
"""

from __future__ import annotations

import os
from functools import lru_cache
from time import sleep

from groq import APIConnectionError, APIStatusError
from langchain_groq import ChatGroq
from langchain_core.utils import convert_to_secret_str
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MAX_JUDGE_ATTEMPTS = 3
ERROR_SUFFIX = "_evaluation_error"


class JudgeResponse(BaseModel):
    """Reject missing fields, coerced scores, out-of-range scores and blank reasons."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    score: int = Field(ge=0, le=3, description="Integer grade from 0 to 3")
    reason: str = Field(min_length=1, description="One sentence explaining the grade")


# ---------------------------------------------------------------------------
# Judge LLM — one instance per process, created on first call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _judge():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY missing from .env — get one free at console.groq.com")
    model = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        api_key=convert_to_secret_str(groq_key),
        model=model,
        temperature=0.1,
        max_tokens=256,
        timeout=30,
        max_retries=0,  # The evaluator owns the retry budget, including parsing failures.
    )


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, TimeoutError, OutputParserException, ValidationError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 409, 429} or error.status_code >= 500
    return False


def _feedback(key, score, comment, attempts, error_type=None):
    failed = score is None
    info = {"status": "error" if failed else "ok", "attempts": attempts}
    if error_type is not None:
        info["error_type"] = error_type
    return {"results": [
        {"key": key, "score": score, "comment": comment, "evaluator_info": info},
        {"key": key + ERROR_SUFFIX, "score": float(failed),
         "comment": "1 = evaluation failed; 0 = answer was scored.",
         "evaluator_info": info},
    ]}


def _make_evaluator(key: str, prompt: str):
    """Return validated scores plus an independent evaluation-error flag.

    Transient API errors and invalid judge responses get up to three attempts,
    with 1s/2s backoff. Configuration/authentication errors fail immediately.
    Missing scores remain None so they cannot be averaged as wrong answers.
    """
    _prompt = ChatPromptTemplate.from_template(prompt)

    def evaluator(run, example):
        if getattr(run, "error", None):
            return _feedback(key, None, "Target run failed; answer was not graded.",
                             attempts=0, error_type="TargetRunError")
        for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
            try:
                judge = _judge().with_structured_output(JudgeResponse, method="json_mode")
                result = (_prompt | judge).invoke({
                    "question": example.inputs.get("question", ""),
                    "reference": (example.outputs or {}).get("answer", "N/A"),
                    "answer":    (run.outputs or {}).get("answer", ""),
                    "context":   (run.outputs or {}).get("context", ""),
                })
                grade = JudgeResponse.model_validate(result)
                return _feedback(key, round(grade.score / 3, 4), grade.reason, attempt)
            except Exception as error:
                if attempt == MAX_JUDGE_ATTEMPTS or not _is_retryable(error):
                    return _feedback(
                        key, None, f"Judge error ({type(error).__name__}): {error}",
                        attempt, error_type=type(error).__name__,
                    )
                sleep(2 ** (attempt - 1))

    evaluator.__name__ = key
    return evaluator


# ---------------------------------------------------------------------------
# Metric 1 — Correctness
# "Is the answer factually correct compared to the reference?"
# Needs: example.outputs["answer"] (ground truth)
# ---------------------------------------------------------------------------

correctness_evaluator = _make_evaluator("correctness", """\
Grade a RAG system answer against a reference answer. Be concise.

Question: {question}
Reference answer: {reference}
RAG answer: {answer}

Scoring scale:
3 = Fully correct. Same meaning as reference, no factual errors.
2 = Mostly correct. Minor omissions, core facts right.
1 = Partially correct. Some facts right but key info wrong or missing.
0 = Incorrect. Contradicts reference or completely off-topic.

Return only a JSON object with exactly two fields:
- "score": an integer from 0 to 3
- "reason": a non-empty string containing one concise sentence
""")


# ---------------------------------------------------------------------------
# Metric 2 — Relevance
# "Does the answer actually address the question asked?"
# No ground truth needed.
# ---------------------------------------------------------------------------

relevance_evaluator = _make_evaluator("relevance", """\
Grade whether a RAG system answer is relevant to the question. Be concise.

Question: {question}
Answer: {answer}

Scoring scale:
3 = Directly and completely answers the question.
2 = Answers the question but incomplete or has minor tangents.
1 = Loosely related but misses the main point.
0 = Does not answer the question at all.

Return only a JSON object with exactly two fields:
- "score": an integer from 0 to 3
- "reason": a non-empty string containing one concise sentence
""")


# ---------------------------------------------------------------------------
# Metric 3 — Groundedness
# "Are the answer's claims supported by the retrieved context?"
# No ground truth needed.
# ---------------------------------------------------------------------------

groundedness_evaluator = _make_evaluator("groundedness", """\
Grade whether a RAG answer is grounded in the retrieved context. Be concise.

Retrieved context:
{context}

Answer: {answer}

Scoring scale:
3 = Every claim supported by context. No hallucination.
2 = Most claims supported; one minor detail not in context.
1 = Some claims supported but notable facts not in context.
0 = Answer contains significant info not present in context.

Return only a JSON object with exactly two fields:
- "score": an integer from 0 to 3
- "reason": a non-empty string containing one concise sentence
""")


# ---------------------------------------------------------------------------
# Metric 4 — Retrieval Relevance
# "Did the retriever fetch chunks useful for this question?"
# No ground truth needed.
# ---------------------------------------------------------------------------

retrieval_relevance_evaluator = _make_evaluator("retrieval_relevance", """\
Grade whether retrieved context chunks are relevant to a question. Be concise.

Question: {question}

Retrieved context:
{context}

Scoring scale:
3 = All or most chunks directly relevant and useful.
2 = Majority relevant; one or two chunks off-topic.
1 = Only a small portion relevant; mostly noisy.
0 = Chunks completely unrelated to the question.

Return only a JSON object with exactly two fields:
- "score": an integer from 0 to 3
- "reason": a non-empty string containing one concise sentence
""")


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

ALL_EVALUATORS = [
    correctness_evaluator,
    relevance_evaluator,
    groundedness_evaluator,
    retrieval_relevance_evaluator,
]
