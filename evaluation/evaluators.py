"""
evaluation/evaluators.py

4 LLM-as-judge evaluators for RAG pipelines.

Scoring: 0-3 scale (normalized to 0.0-1.0 for LangSmith)
    0 = completely wrong / irrelevant / hallucinated
    1 = partially correct / somewhat relevant
    2 = mostly correct / relevant with minor issues
    3 = fully correct / highly relevant / well-grounded

Judge LLM: Groq model from the JUDGE_MODEL env var (default below in _judge).
Uses GROQ_API_KEY.

Failures: if the judge call errors or no "Score: N" can be parsed, the score is
None (not 0), so LangSmith leaves that row out of the average instead of
counting it as a bad answer. The reason is kept in the comment.
If the RAG chain itself crashed, the judge is not called and the score is None.

Metrics:
    correctness         -- does answer match the reference? (needs ground truth)
    relevance           -- does answer actually address the question?
    groundedness        -- are all claims backed by retrieved context?
    retrieval_relevance -- are retrieved chunks on-topic for the question?
    evidence_recall     -- did retrieval find the labeled evidence pages? (no LLM)
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.utils import convert_to_secret_str
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        temperature=0,    # same input -> same grade, as far as the API allows
        max_tokens=1024,  # room for a <think> block before "Score: N"
    )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SCORE = re.compile(r"^[ \t*_]*score[ \t*_]*[:=][ \t*_]*([0-3])\b(?!\.\d|[ \t]*/)", re.I | re.M)


def parse_score(text: str) -> int | None:
    """Return the 0-3 score from the judge reply, or None if there is none."""
    text = _THINK.sub("", text)
    if "<think>" in text.lower():   # thinking was cut off: no real answer
        return None
    match = _SCORE.search(text)   # first "Score:" line; later ones are in the reason
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

def _make_evaluator(key: str, prompt: str):
    """
    Build a LangSmith-compatible evaluator from a prompt template.

    Prompt must instruct the LLM to output:
        Score: <0|1|2|3>
        Reason: <one sentence>

    score normalised to 0.0-1.0 (divided by 3) for LangSmith.
    On judge error or parse failure the score is None — never raises.
    """
    _prompt = ChatPromptTemplate.from_template(prompt)

    def evaluator(run, example):
        if getattr(run, "error", None):
            return {"key": key, "score": None, "comment": f"TARGET_ERROR: {run.error[:300]}"}

        try:
            chain = _prompt | _judge() | StrOutputParser()
            result = chain.invoke({
                "question": example.inputs.get("question", ""),
                "reference": (example.outputs or {}).get("answer", "N/A"),
                "answer":    (run.outputs or {}).get("answer", ""),
                "context":   (run.outputs or {}).get("context", ""),
            }).strip()
        except Exception as e:
            return {"key": key, "score": None, "comment": f"JUDGE_ERROR: {e}"}

        score_raw = parse_score(result)
        if score_raw is None:
            return {"key": key, "score": None, "comment": f"JUDGE_PARSE_ERROR: {result[-500:]}"}

        return {
            "key":     key,
            "score":   round(score_raw / 3, 4),
            "comment": _THINK.sub("", result).strip(),
        }

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

Output exactly:
Score: <0|1|2|3>
Reason: <one sentence>
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

Output exactly:
Score: <0|1|2|3>
Reason: <one sentence>
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

Output exactly:
Score: <0|1|2|3>
Reason: <one sentence>
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

Output exactly:
Score: <0|1|2|3>
Reason: <one sentence>
""")


# ---------------------------------------------------------------------------
# Metric 5 — Evidence recall (no LLM)
# "Did the retriever find the pages that hold the answer?"
# Needs: example.outputs["evidence"] and run.outputs["pages"]
# ---------------------------------------------------------------------------

NOT_APPLICABLE = "NOT_APPLICABLE"


def evidence_recall_evaluator(run, example):
    """Share of evidence items found in the retrieved chunks (0.0-1.0).

    An evidence item counts as found if any retrieved chunk comes from its
    page or one of its "also" pages. Unanswerable questions have no evidence,
    so their score is None with a NOT_APPLICABLE comment.
    """
    key = "evidence_recall"
    if getattr(run, "error", None):
        return {"key": key, "score": None, "comment": f"TARGET_ERROR: {run.error[:300]}"}
    evidence = (example.outputs or {}).get("evidence") or []
    if not evidence:
        return {"key": key, "score": None, "comment": f"{NOT_APPLICABLE}: no evidence"}

    retrieved = {(p["source"], p["page"]) for p in (run.outputs or {}).get("pages", [])}
    missed = []
    for ev in evidence:
        wanted = {(ev["source"], page) for page in [ev["page"], *ev.get("also", [])]}
        if not retrieved & wanted:
            missed.append(f"{ev['source']} page {ev['page']} ({ev['phrase']!r})")
    found = len(evidence) - len(missed)
    comment = f"found {found}/{len(evidence)}" + (f"; missed {', '.join(missed)}" if missed else "")
    return {"key": key, "score": round(found / len(evidence), 4), "comment": comment}


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

ALL_EVALUATORS = [
    correctness_evaluator,
    relevance_evaluator,
    groundedness_evaluator,
    retrieval_relevance_evaluator,
    evidence_recall_evaluator,
]
