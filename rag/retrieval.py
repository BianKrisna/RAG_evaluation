"""Question-to-documents building blocks."""
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from typing import List

class HybridRetriever(BaseRetriever):
    """Combines a dense (FAISS) and sparse (BM25) retriever via RRF."""

    dense: BaseRetriever = Field(description="Dense vector retriever")
    sparse: BaseRetriever = Field(description="BM25 sparse retriever")
    dense_weight: float = Field(default=0.5, description="Weight for dense results")
    sparse_weight: float = Field(default=0.5, description="Weight for sparse results")
    top_k: int = Field(default=4, description="Number of final docs to return")
    rrf_k: int = Field(default=60, description="RRF constant (dampens top-rank advantage)")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Retrieve from both sides independently.
        dense_docs = self.dense.invoke(query)
        sparse_docs = self.sparse.invoke(query)

        # Build a map: doc page_content → (Document, fused_score).
        # page_content used as dedup key (same chunk from both retrievers).
        scores: dict[str, tuple[Document, float]] = {}

        def apply_rrf(docs: List[Document], weight: float) -> None:
            for rank, doc in enumerate(docs, start=1):
                # how rrf works
                key = doc.page_content
                rrf_score = weight / (rank + self.rrf_k)
                if key in scores:
                    scores[key] = (scores[key][0], scores[key][1] + rrf_score)
                else:
                    scores[key] = (doc, rrf_score)

        apply_rrf(dense_docs, self.dense_weight)
        apply_rrf(sparse_docs, self.sparse_weight)

        # Sort by fused score descending, return top_k.
        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]



def dense_retrieval(vectorstore, **search_kwargs):
    return vectorstore.as_retriever(**({"search_kwargs": search_kwargs} if search_kwargs else {}))


def sparse_retrieval(documents, k=4):
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever
