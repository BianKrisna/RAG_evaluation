"""
eval_baseline.py — Evaluate the baseline RAG with LangSmith.

Run:
    source venv/bin/activate
    python eval_baseline.py

Required env vars (add to .env):
    LANGSMITH_API_KEY   — from smith.langchain.com
    LANGSMITH_TRACING=true
    OPENAI_API_KEY      — used by judge LLM (gpt-4o-mini by default)
    HF_API              — already set; used by baseline RAG

Optional:
    JUDGE_MODEL         — override judge model (default: gpt-4o-mini)
    LANGSMITH_PROJECT   — LangSmith project name (default: "RAG_langchain")
"""

import os

# Prevents malloc double-free crash on Apple Silicon from HF tokenizer threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "RAG_langchain")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from evaluation import create_or_get_dataset, evaluate_rag, ALL_EVALUATORS


def build_baseline_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        "my_vector_db", embeddings, allow_dangerous_deserialization=True
    )

    hf_token = os.getenv("HF_API")
    if not hf_token:
        raise RuntimeError("HF_API missing from .env")

    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            huggingfacehub_api_token=hf_token,
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            task="text-generation",
            max_new_tokens=512,
        )
    )

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

    qa_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    return create_retrieval_chain(retriever, qa_chain)


# ── Evaluation dataset ───────────────────────────────────────────────────────
# Edit these to match your document domain.
# "answer" = ground truth (needed for correctness metric).
# Leave "answer" empty if no ground truth — correctness will score 0.
EVAL_EXAMPLES = [
    {
        "question": "What is the main purpose of the system described in the document?",
        "answer": "The main purpose of the Internet is to act as a networking infrastructure that provides services to distributed applications (such as web, email, video streaming, and social networks) and allows end systems (hosts) to exchange data reliably and efficiently across the globe.",
    },
    {
        "question": "What are the key components mentioned in the document?",
        "answer": "The key components of the network include end systems (or hosts, like PCs, smartphones, and servers), communication links (such as coaxial cable, copper wire, optical fiber, and radio spectrum), packet switches (primarily routers and link-layer switches), and Internet Service Providers (ISPs) that connect them all together.",
    },
    {
        "question": "What protocols or standards does the document reference?",
        "answer": "The text references Internet standards developed by the IETF (Internet Engineering Task Force) documented as RFCs (Requests for Comments). Key protocols include HTTP, SMTP, and DNS at the application layer; TCP and UDP at the transport layer; IP, OSPF, and BGP at the network layer; and Ethernet and IEEE 802.11 (Wi-Fi) at the link layer.",
    },
]

DATASET_NAME = "baseline-rag-eval"


if __name__ == "__main__":
    print("Building baseline chain...")
    chain = build_baseline_chain()

    print("Sanity-check invoke...")
    test = chain.invoke({"input": "test"})
    print("Chain OK, answer snippet:", str(test.get("answer", ""))[:80])

    print("Creating/checking LangSmith dataset...")
    create_or_get_dataset(DATASET_NAME, EVAL_EXAMPLES, overwrite=True)

    print("Running evaluation (this calls LangSmith + judge LLM)...")
    results = evaluate_rag(
        rag_chain=chain,
        dataset_name=DATASET_NAME,
        experiment_prefix="baseline",
        evaluators=ALL_EVALUATORS,
        max_concurrency=1,
    )

    print("\nView full results at: https://smith.langchain.com")
