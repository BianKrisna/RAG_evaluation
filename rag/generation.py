"""Replaceable prompt, model and document-generation blocks."""
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise.\n\n"
)


def build_prompt(system_prompt=SYSTEM_PROMPT):
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt + "{context}"), ("human", "{input}"),
    ])


def build_llm(repo_id="meta-llama/Llama-3.1-8B-Instruct", temperature=0.1, max_new_tokens=512):
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    token = os.getenv("HF_API")
    if not token:
        raise RuntimeError("HF_API missing from .env")
    return ChatHuggingFace(llm=HuggingFaceEndpoint(
        huggingfacehub_api_token=token, repo_id=repo_id, temperature=temperature,
        task="text-generation", max_new_tokens=max_new_tokens,
    ))


def build_generation(llm=None, prompt=None, **stuff_options):
    """dict(input, context: list[Document]) -> str; accepts parser/format overrides."""
    return create_stuff_documents_chain(
        llm if llm is not None else build_llm(),
        prompt if prompt is not None else build_prompt(), **stuff_options,
    )
