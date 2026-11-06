"""Composable RAG blocks; no resources are created at import time."""
from .architectures import ARCHITECTURES, assemble, baseline, hybrid
from .generation import build_generation, build_llm, build_prompt
from .preparation import Resources, prepare_resources
from .retrieval import HybridRetriever, dense_retrieval, sparse_retrieval
