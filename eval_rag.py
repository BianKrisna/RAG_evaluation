"""Evaluate one RAG architecture or compare several on a shared LangSmith dataset.

    python eval_rag.py --architectures baseline
    python eval_rag.py --architectures baseline hybrid

Requires HF_API, LANGSMITH_API_KEY and GROQ_API_KEY; JUDGE_MODEL is optional.
"""
import argparse


def configure_environment():
    import os
    from dotenv import load_dotenv

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_dotenv()
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "RAG_langchain")


def main(argv=None):
    configure_environment()
    from langsmith import Client
    from rag import ARCHITECTURES, prepare_resources
    from rag.architectures import ARCHITECTURE_SETTINGS
    from evaluation import create_or_get_dataset
    from evaluation.examples import EVAL_EXAMPLES, DATASET_NAME
    from evaluation.comparison import compare_architectures

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=["baseline", "hybrid"])
    parser.add_argument("--dataset", default=DATASET_NAME)
    args = parser.parse_args(argv)
    client = Client()
    create_or_get_dataset(args.dataset, EVAL_EXAMPLES, client=client)
    resources = prepare_resources()
    chains = {name: ARCHITECTURES[name](resources) for name in args.architectures}
    common = {"chunk_size": 1024, "chunk_overlap": 100,
              "embedding_model": "all-MiniLM-L6-v2", "normalize_embeddings": True,
              "model": "meta-llama/Llama-3.1-8B-Instruct", "temperature": 0.1,
              "max_new_tokens": 512}
    settings = {name: {**common, **ARCHITECTURE_SETTINGS.get(name, {})} for name in chains}
    results = compare_architectures(chains, args.dataset, settings=settings, client=client)
    for name, result in results.items():
        experiment = client.read_project(project_name=result.experiment_name)
        print(f"{name}: {experiment.url}")


if __name__ == "__main__":
    main()
