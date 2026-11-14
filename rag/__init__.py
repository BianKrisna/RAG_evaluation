"""Composable RAG blocks; no resources are created at import time."""
from .architectures import ARCHITECTURES, assemble, baseline, hybrid, reranked
from .generation import build_generation, build_llm, build_prompt
from .evidence import prepare_evidence, resolve_citations
from .preparation import Resources, prepare_resources
from .retrieval import HybridRetriever, RerankingRetriever, cross_encoder, dense_retrieval, sparse_retrieval
