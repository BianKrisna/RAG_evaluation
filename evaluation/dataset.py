"""
dataset.py — LangSmith dataset helpers.

create_or_get_dataset(name, examples) creates (or reuses) a LangSmith dataset.

The real dataset name is "<name>-<fingerprint>", where the fingerprint is a
short hash of the examples. Editing the examples gives a new fingerprint, so a
new dataset is created and old experiments stay attached to the old one. An
existing dataset is only reused when its content is identical.

examples: list of dicts (see evaluation/examples.py), each with keys:
    "id", "type", "question", "answer", "answerable", "evidence"
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
    Create or reuse the LangSmith dataset for exactly these examples.

    Args:
        name:      Base dataset name; the fingerprint is appended.
        examples:  Benchmark examples.
        client:    LangSmith Client (created from env vars if None).
        overwrite: Delete + recreate the dataset if it already exists.

    Returns:
        The versioned dataset name (use this as `data` in langsmith.evaluate()).
    """
    from .examples import benchmark_fingerprint

    client = client or Client()
    fingerprint = benchmark_fingerprint(examples)
    full_name = f"{name}-{fingerprint}"

    existing = [d for d in client.list_datasets(dataset_name=full_name)]

    if existing:
        if overwrite:
            client.delete_dataset(dataset_id=existing[0].id)
        else:
            # Same fingerprint means same content, unless an earlier upload was cut short.
            uploaded = sum(1 for _ in client.list_examples(dataset_id=existing[0].id))
            if uploaded != len(examples):
                raise RuntimeError(
                    f"Dataset '{full_name}' has {uploaded} examples but the benchmark has "
                    f"{len(examples)}. Call create_or_get_dataset(..., overwrite=True) to rebuild it."
                )
            print(f"[dataset] Reusing existing dataset '{full_name}' ({existing[0].id})")
            return full_name

    dataset = client.create_dataset(
        full_name,
        description=f"{name} benchmark, {len(examples)} examples, fingerprint {fingerprint}",
        metadata={"fingerprint": fingerprint, "num_examples": len(examples)},
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": ex["question"]},
                "outputs": {
                    "answer": ex.get("answer", ""),
                    "answerable": ex.get("answerable", True),
                    "evidence": ex.get("evidence", []),
                },
                "metadata": {"example_id": ex.get("id"), "type": ex.get("type")},
            }
            for ex in examples
        ],
    )
    print(f"[dataset] Created dataset '{full_name}' with {len(examples)} examples.")
    return full_name
