"""Benchmark examples for the RAG evaluation.

The data lives in benchmark.json (next to this file). Each example has:

    id          unique short id, e.g. "f01"
    type        factual | keyword | paraphrase | multi_chunk | unanswerable
    question    what is sent to the RAG chain
    answer      reference answer (used by the correctness judge)
    answerable  False for questions the document does not cover
    evidence    where the answer lives: [{"source", "page", "phrase"}]
                page is the 0-based page index, the same number PyMuPDFLoader
                puts in doc.metadata["page"]; phrase is a short anchor that
                appears on that page
    absent_terms  (unanswerable only) terms that must NOT appear in the corpus

Check the benchmark against the real PDF (no API keys needed):

    python -m evaluation.examples
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

BENCHMARK_PATH = Path(__file__).with_name("benchmark.json")
DATASET_NAME = "RAG-eval"
EXAMPLE_TYPES = ("factual", "keyword", "paraphrase", "multi_chunk", "unanswerable")


def load_benchmark(path=BENCHMARK_PATH) -> dict:
    """Read benchmark.json and fail early if its structure is wrong."""
    with open(path, encoding="utf-8") as f:
        benchmark = json.load(f)
    problems = validate_benchmark(benchmark)
    if problems:
        raise ValueError("Invalid benchmark:\n  " + "\n  ".join(problems))
    return benchmark


def validate_benchmark(benchmark: dict) -> list[str]:
    """Structure checks that need no corpus. Returns a list of problems."""
    problems = []
    examples = benchmark.get("examples") or []
    if not examples:
        return ["benchmark has no examples"]
    total_pages = (benchmark.get("corpus") or {}).get("pages")
    seen_ids, seen_questions = set(), set()
    for i, ex in enumerate(examples):
        tag = ex.get("id") or f"#{i}"
        if not ex.get("id"):
            problems.append(f"{tag}: missing id")
        elif ex["id"] in seen_ids:
            problems.append(f"{tag}: duplicate id")
        seen_ids.add(ex.get("id"))
        question = (ex.get("question") or "").strip()
        if not question:
            problems.append(f"{tag}: empty question")
        elif question.lower() in seen_questions:
            problems.append(f"{tag}: duplicate question")
        seen_questions.add(question.lower())
        if not (ex.get("answer") or "").strip():
            problems.append(f"{tag}: empty answer")
        if ex.get("type") not in EXAMPLE_TYPES:
            problems.append(f"{tag}: unknown type {ex.get('type')!r}")
        answerable = ex.get("answerable")
        if not isinstance(answerable, bool):
            problems.append(f"{tag}: 'answerable' must be true or false")
        elif answerable != (ex.get("type") != "unanswerable"):
            problems.append(f"{tag}: 'answerable' does not match type {ex.get('type')!r}")
        evidence = ex.get("evidence") or []
        if answerable and not evidence:
            problems.append(f"{tag}: answerable example needs evidence")
        if answerable is False and evidence:
            problems.append(f"{tag}: unanswerable example must not have evidence")
        if answerable is False and not ex.get("absent_terms"):
            problems.append(f"{tag}: unanswerable example needs absent_terms")
        if ex.get("type") == "multi_chunk" and len({(e.get("source"), e.get("page")) for e in evidence}) < 2:
            problems.append(f"{tag}: multi_chunk example needs evidence on at least two pages")
        for ev in evidence:
            if not ev.get("source") or not (ev.get("phrase") or "").strip():
                problems.append(f"{tag}: evidence needs 'source' and 'phrase'")
            page = ev.get("page")
            if not isinstance(page, int) or page < 0 or (total_pages and page >= total_pages):
                problems.append(f"{tag}: evidence page {page!r} is out of range")
    return problems


def benchmark_fingerprint(examples) -> str:
    """Short hash of the examples. Changes whenever any example changes."""
    canonical = json.dumps(sorted(examples, key=lambda ex: ex["id"]),
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def normalize_text(text: str) -> str:
    """Make PDF text comparable: fix ligatures, join hyphenated line breaks,
    unify quotes and dashes, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    for old, new in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                     ("–", "-"), ("—", "-")):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).lower().strip()


def verify_corpus(benchmark: dict, corpus_dir="./src") -> list[str]:
    """Check the benchmark against the real PDF files. Returns a list of problems.

    - every evidence phrase must appear on its page
    - no absent_term of an unanswerable example may appear anywhere
    """
    import pymupdf

    problems, pages_by_source = [], {}

    def pages(source):
        if source not in pages_by_source:
            path = Path(corpus_dir) / source
            if not path.exists():
                pages_by_source[source] = None
                problems.append(f"corpus file not found: {path}")
            else:
                with pymupdf.open(path) as pdf:
                    pages_by_source[source] = [normalize_text(page.get_text()) for page in pdf]
        return pages_by_source[source]

    corpus = benchmark.get("corpus") or {}
    main_pages = pages(corpus["source"]) if corpus.get("source") else None
    if main_pages is not None and corpus.get("pages") not in (None, len(main_pages)):
        problems.append(f"corpus has {len(main_pages)} pages, benchmark.json says {corpus['pages']}")

    for ex in benchmark["examples"]:
        for ev in ex.get("evidence") or []:
            doc = pages(ev["source"])
            if doc is None:
                continue
            if ev["page"] >= len(doc):
                problems.append(f"{ex['id']}: page {ev['page']} does not exist in {ev['source']}")
            elif normalize_text(ev["phrase"]) not in doc[ev["page"]]:
                problems.append(f"{ex['id']}: phrase {ev['phrase']!r} not found on page {ev['page']}")
        for term in ex.get("absent_terms") or []:
            pattern = re.compile(r"\b" + re.escape(normalize_text(term)) + r"\b")
            for doc in pages_by_source.values():
                hit = next((n for n, text in enumerate(doc or []) if pattern.search(text)), None)
                if hit is not None:
                    problems.append(f"{ex['id']}: term {term!r} appears on page {hit}, "
                                    "so the question may be answerable")
                    break
    return problems


_BENCHMARK = load_benchmark()
EVAL_EXAMPLES = _BENCHMARK["examples"]


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate benchmark.json against the corpus.")
    parser.add_argument("--corpus", default="./src", help="folder with the source PDF files")
    args = parser.parse_args(argv)
    counts = Counter(ex["type"] for ex in EVAL_EXAMPLES)
    print(f"{len(EVAL_EXAMPLES)} examples, fingerprint {benchmark_fingerprint(EVAL_EXAMPLES)}")
    for name in EXAMPLE_TYPES:
        print(f"  {name:<13}{counts.get(name, 0)}")
    problems = verify_corpus(_BENCHMARK, args.corpus)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nAll evidence phrases found on their pages; no absent term appears in the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
