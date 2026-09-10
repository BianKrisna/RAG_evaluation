"""
dataset.py — LangSmith dataset helpers.

create_or_get_dataset(name, examples) creates (or reuses) a LangSmith dataset.

examples: list of dicts, each with keys:
    "question"  (str)           — required
    "answer"    (str, optional) — reference/ground-truth answer
                                  (needed for correctness metric)
"""

from __future__ import annotations
from typing import Any

from langsmith import Client


def create_or_get_dataset(
    name: str,
    examples: list[dict[str, Any]],
    client: Client | None = None,
    overwrite: bool = False,
) -> str:
    """
    Create or reuse a LangSmith dataset.

    Returns the dataset name (usable as `data` in langsmith.evaluate()).

    Args:
        name:      Dataset name in LangSmith.
        examples:  List of {"question": ..., "answer": ...} dicts.
                   "answer" key is optional (needed only for correctness).
        client:    LangSmith Client (created from env vars if None).
        overwrite: Delete + recreate the dataset if it already exists.

    Returns:
        Dataset name string.
    """
    client = client or Client()

    existing = [d for d in client.list_datasets(dataset_name=name)]

    if existing:
        if overwrite:
            client.delete_dataset(dataset_id=existing[0].id)
        else:
            print(f"[dataset] Reusing existing dataset '{name}' ({existing[0].id})")
            return name

    dataset = client.create_dataset(name)
    inputs = [{"question": ex["question"]} for ex in examples]
    outputs = [{"answer": ex.get("answer", "")} for ex in examples]
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    print(f"[dataset] Created dataset '{name}' with {len(examples)} examples.")
    return name
