"""Sequential experiments over one materialized dataset snapshot."""
from uuid import uuid4
from datetime import datetime, timezone
from langsmith import Client
from .runner import evaluate_rag


def compare_architectures(chains, dataset_name, settings=None, client=None):
    client = client or Client()
    snapshot_time = datetime.now(timezone.utc)
    examples = list(client.list_examples(dataset_name=dataset_name, as_of=snapshot_time))
    if not examples:
        raise ValueError("Comparison dataset is empty")
    comparison_id = str(uuid4())
    results = {}
    for name, chain in chains.items():
        results[name] = evaluate_rag(
            chain, dataset_name, experiment_prefix=f"{name}-{comparison_id}",
            client=client, max_concurrency=1, data=examples,
            metadata={"architecture": name, "comparison_id": comparison_id,
                      "dataset_snapshot": snapshot_time.isoformat(),
                      "settings": (settings or {}).get(name, {})},
        )
    return results
