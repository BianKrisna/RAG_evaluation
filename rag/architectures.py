"""Named presets composed from ordinary LangChain Runnables."""
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .evidence import prepare_evidence, resolve_citations
from .generation import build_generation
from .retrieval import (
    CROSS_ENCODER_MODEL, HybridRetriever, cross_encoder, dense_retrieval, sparse_retrieval,
)


def assemble(retrieval, generation):
    """Compose retrieval/generation and return text, evidence, and citation validation."""
    return (
        RunnablePassthrough.assign(
            context=itemgetter("input") | retrieval.with_config(run_name="retrieve_documents")
            | RunnableLambda(prepare_evidence)
        ).assign(answer=generation)
        | RunnableLambda(lambda result: {
            **result, **resolve_citations(result["answer"], result["context"]),
        })
    ).with_config(run_name="retrieval_chain")


def baseline(resources, llm=None, prompt=None):
    return assemble(dense_retrieval(resources.vectorstore), build_generation(llm, prompt))


def hybrid(resources, llm=None, prompt=None):
    retrieval = HybridRetriever(
        dense=dense_retrieval(resources.vectorstore, k=4),
        sparse=sparse_retrieval(resources.documents, k=4),
        dense_weight=0.5, sparse_weight=0.5, top_k=4,
    )
    return assemble(retrieval, build_generation(llm, prompt))


def reranked(resources, llm=None, prompt=None):
    retrieval = HybridRetriever(
        dense=dense_retrieval(resources.vectorstore, k=20),
        sparse=sparse_retrieval(resources.documents, k=20),
        dense_weight=0.5, sparse_weight=0.5, top_k=20,
    )
    return assemble(cross_encoder(retrieval, top_k=4), build_generation(llm, prompt))


# Each factory accepts (resources, llm=None, prompt=None). Add custom factories here.
ARCHITECTURES = {"baseline": baseline, "hybrid": hybrid, "reranked": reranked}
ARCHITECTURE_SETTINGS = {
    "baseline": {"retrieval": "dense", "k": 4},
    "hybrid": {"retrieval": "dense+BM25+RRF", "dense_k": 4, "sparse_k": 4,
               "top_k": 4, "dense_weight": 0.5, "sparse_weight": 0.5, "rrf_k": 60},
    "reranked": {"retrieval": "dense+BM25+RRF+cross-encoder", "dense_k": 20, "sparse_k": 20,
                 "candidate_k": 20, "top_k": 4, "dense_weight": 0.5, "sparse_weight": 0.5,
                 "rrf_k": 60, "reranker_model": CROSS_ENCODER_MODEL},
}
