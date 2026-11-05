"""Explicit preparation: importing this module does not load models or documents."""
from dataclasses import dataclass
from typing import Any


def load_documents(path="./src"):
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.document_loaders.directory import DirectoryLoader
    return DirectoryLoader(path=path, loader_cls=PyMuPDFLoader).load()


def split_documents(documents, chunk_size=1024, chunk_overlap=100):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(documents)


def build_embeddings(model_name="all-MiniLM-L6-v2"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def build_vectorstore(documents, embeddings):
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(documents, embeddings)


def load_vectorstore(path="my_vector_db", embeddings=None):
    from langchain_community.vectorstores import FAISS
    # Same trusted local index loading policy as the legacy evaluator.
    if embeddings is None:
        embeddings = build_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


@dataclass
class Resources:
    documents: list
    vectorstore: Any


def prepare_resources(path="./src", save_path=None):
    documents = split_documents(load_documents(path))
    vectorstore = build_vectorstore(documents, build_embeddings())
    if save_path is not None:
        vectorstore.save_local(save_path)
    return Resources(documents, vectorstore)
