"""Named presets composed from ordinary LangChain Runnables."""
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from .generation import build_generation
from .retrieval import HybridRetriever, dense_retrieval, sparse_retrieval


def assemble(retrieval, generation):
    """retrieval: str -> documents; generation: dict -> answer."""
    return RunnablePassthrough.assign(
        context=itemgetter("input") | retrieval.with_config(run_name="retrieve_documents")
    ).assign(answer=generation).with_config(run_name="retrieval_chain")


def baseline(resources, llm=None, prompt=None):
    return assemble(dense_retrieval(resources.vectorstore), build_generation(llm, prompt))


def hybrid(resources, llm=None, prompt=None):
    retrieval = HybridRetriever(
        dense=dense_retrieval(resources.vectorstore, k=4),
        sparse=sparse_retrieval(resources.documents, k=4),
        dense_weight=0.5, sparse_weight=0.5, top_k=4,
    )
    return assemble(retrieval, build_generation(llm, prompt))


# Each factory accepts (resources, llm=None, prompt=None). Add custom factories here.
ARCHITECTURES = {"baseline": baseline, "hybrid": hybrid}
ARCHITECTURE_SETTINGS = {
    "baseline": {"retrieval": "dense", "k": 4},
    "hybrid": {"retrieval": "dense+BM25+RRF", "dense_k": 4, "sparse_k": 4,
               "top_k": 4, "dense_weight": 0.5, "sparse_weight": 0.5, "rrf_k": 60},
}
