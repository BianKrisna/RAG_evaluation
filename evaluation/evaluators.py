"""
evaluation/evaluators.py

4 LLM-as-judge evaluators for RAG pipelines.

Scoring: 0-3 scale (normalized to 0.0-1.0 for LangSmith)
    0 = completely wrong / irrelevant / hallucinated
    1 = partially correct / somewhat relevant
    2 = mostly correct / relevant with minor issues
    3 = fully correct / highly relevant / well-grounded

Judge LLM: meta-llama/Llama-3.1-8B-Instruct via HuggingFace Inference API.
Override with JUDGE_MODEL env var. Uses HF_API key.

Metrics:
    correctness         -- does answer match the reference? (needs ground truth)
    relevance           -- does answer actually address the question?
    groundedness        -- are all claims backed by retrieved context?
    retrieval_relevance -- are retrieved chunks on-topic for the question?
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Judge LLM — one instance per process, created on first call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _judge():
    hf_token = os.getenv("HF_API")
    if not hf_token:
        raise RuntimeError("HF_API missing from .env")
    model = os.getenv("JUDGE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    endpoint = HuggingFaceEndpoint(
        huggingfacehub_api_token=hf_token,
        repo_id=model,
        temperature=0.1,
        task="text-generation",
        max_new_tokens=128,   # short — only needs "Score: N\nReason: ..."
    )
    return ChatHuggingFace(llm=endpoint)


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
    Falls back to 0 on parse failure — never raises.
    """
    _prompt = ChatPromptTemplate.from_template(prompt)

    def evaluator(run, example):
        try:
            chain = _prompt | _judge() | StrOutputParser()
            result = chain.invoke({
                "question": example.inputs.get("question", ""),
                "reference": (example.outputs or {}).get("answer", "N/A"),
                "answer":    (run.outputs or {}).get("answer", ""),
                "context":   (run.outputs or {}).get("context", ""),
            }).strip()
        except Exception as e:
            return {"key": key, "score": 0.0, "comment": f"Judge error: {e}"}

        # Parse "Score: N" line — robust against preamble/postamble
        score_raw = 0
        for line in result.splitlines():
            if line.lower().startswith("score:"):
                try:
                    score_raw = int(line.split(":", 1)[1].strip()[0])
                    score_raw = max(0, min(3, score_raw))
                except (ValueError, IndexError):
                    score_raw = 0
                break

        return {
            "key":     key,
            "score":   round(score_raw / 3, 4),
            "comment": result,
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
# Bundle
# ---------------------------------------------------------------------------

ALL_EVALUATORS = [
    correctness_evaluator,
    relevance_evaluator,
    groundedness_evaluator,
    retrieval_relevance_evaluator,
]
