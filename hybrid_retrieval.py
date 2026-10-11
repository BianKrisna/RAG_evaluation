from langchain_community.document_loaders import directory, PyMuPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from typing import List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os
from dotenv import load_dotenv

load_dotenv()

# ── 1. Load documents ────────────────────────────────────────────────────────
# DirectoryLoader walks ./src and hands each PDF to PyMuPDFLoader.
# PyMuPDF is fast and preserves text layout better than most PDF parsers.
loader = directory.DirectoryLoader(path="./src", loader_cls=PyMuPDFLoader)
loaded_files = loader.load()

# ── 2. Split into chunks ─────────────────────────────────────────────────────
# chunk_size=1024 tokens keeps enough context per chunk.
# chunk_overlap=100 prevents answers that span a chunk boundary from being lost.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100
)
splitted_files = text_splitter.split_documents(loaded_files)

# ── 3. Dense retriever (FAISS + embeddings) ──────────────────────────────────
# Converts each chunk to a vector; retrieval finds semantically similar chunks.
# "all-MiniLM-L6-v2" is small, fast, and good for English text.
# normalize_embeddings=True makes cosine similarity == dot product (faster FAISS search).
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.from_documents(splitted_files, embeddings)
vectorstore.save_local("my_vector_db")

# k=4: return top-4 chunks from the dense side.
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── 4. Sparse retriever (BM25) ───────────────────────────────────────────────
# BM25 is a classical keyword-matching algorithm (no embeddings).
# Strong at exact-term matches that dense vectors sometimes miss
# (rare proper nouns, codes, version numbers, etc.).
# k=4: return top-4 chunks from the sparse side.
bm25_retriever = BM25Retriever.from_documents(splitted_files)
bm25_retriever.k = 4

# ── 5. Hybrid retriever (Reciprocal Rank Fusion) ─────────────────────────────
# EnsembleRetriever isn't available in this langchain version, so we implement
# Reciprocal Rank Fusion (RRF) manually as a custom BaseRetriever.
#
# How RRF works:
#   Each retriever returns a ranked list of documents.
#   For every document, accumulate:  score += weight / (rank + rrf_k)
#   where rank starts at 1, rrf_k=60 dampens the effect of top ranks.
#   Documents from both retrievers are merged; duplicates get their scores summed.
#   Final list is sorted descending by fused score, top-k returned.
#
# Why this beats picking just one retriever:
#   Dense catches semantic paraphrases; BM25 catches exact keywords.
#   A document that ranks highly in BOTH gets a much higher fused score.

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


# Tune weights toward dense (e.g. dense_weight=0.7) for general semantic queries,
# or toward sparse (e.g. sparse_weight=0.7) for exact-keyword-heavy queries.
hybrid_retriever = HybridRetriever(
    dense=dense_retriever,
    sparse=bm25_retriever,
    dense_weight=0.5,
    sparse_weight=0.5,
    top_k=4,
)

# ── 6. LLM ───────────────────────────────────────────────────────────────────
# Same model and config as baseline.py.
huggingface_token = os.getenv("HF_API")
if not huggingface_token:
    raise RuntimeError(
        "Hugging Face token is missing. Set HF_API in .env before running."
    )

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        huggingfacehub_api_token=huggingface_token,
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        task="text-generation",
        max_new_tokens=512
    )
)

# ── 7. Prompt ────────────────────────────────────────────────────────────────
# {context} is filled with the retrieved chunks (formatted by create_stuff_documents_chain).
# {input} is filled with the user's question.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise.\n\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "{context}"),
    ("human", "{input}"),
])

# ── 8. Chain ─────────────────────────────────────────────────────────────────
# create_stuff_documents_chain: concatenates all retrieved chunks into {context},
# then passes them with the prompt to the LLM.
# create_retrieval_chain: wires retriever → document chain end-to-end.
# Input key: "input" (the question). Output key: "answer".
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(hybrid_retriever, question_answer_chain)

# ── 9. Run ───────────────────────────────────────────────────────────────────
question = input("What is your question? ")
response = rag_chain.invoke({"input": question})
print(response["answer"])
