"""Deterministic source identity, passage formatting, and citation resolution."""
import hashlib
import json
import posixpath
import re
from pathlib import PurePosixPath

from langchain_core.documents import Document


ABSTENTION_ANSWER = "I don't know based on the provided documents."
CITATION_PATTERN = re.compile(r"\[(chunk_[^\]]*)\]")
EVIDENCE_TEMPLATE = "[{chunk_id}]\nSource: {source}\nPDF page: {page_display}\n{page_content}"


def _identity(prefix, *parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def prepare_evidence(documents: list[Document]) -> list[Document]:
    """Copy documents with stable IDs; never overwrite loader page indices.

    Document identity is its normalized source location. Chunk identity includes
    that location, zero-based page, character offset, and exact chunk content.
    Missing provenance is explicitly unknown, never assigned a guessed page.
    """
    prepared = []
    for doc in documents:
        metadata = dict(doc.metadata)
        source = str(metadata.get("source") or metadata.get("file_path") or "unknown")
        if "://" not in source:
            source = posixpath.normpath(source.replace("\\", "/"))
        page = metadata.get("page")
        if page is not None and (type(page) is not int or page < 0):
            raise ValueError("Document page must be a non-negative, zero-based integer or None")
        start_index = metadata.get("start_index")
        if start_index is not None and (type(start_index) is not int or start_index < 0):
            raise ValueError("Document start_index must be a non-negative integer or None")
        document_id = _identity("doc_", source)
        chunk_id = _identity("chunk_", document_id, page, start_index, doc.page_content)
        page_number = page + 1 if page is not None else None
        metadata.update({
            "document_id": document_id,
            "chunk_id": chunk_id,
            "source": source,
            "filename": PurePosixPath(source).name,
            "page": page,
            "page_number": page_number,
            "page_display": str(page_number) if page_number is not None else "unknown",
            "start_index": start_index,
        })
        prepared.append(Document(id=chunk_id, page_content=doc.page_content, metadata=metadata))
    return prepared


def format_evidence(document: Document) -> str:
    """Use the same evidence header in model context and evaluation context."""
    doc = prepare_evidence([document])[0]
    return EVIDENCE_TEMPLATE.format(**{**doc.metadata, "page_content": doc.page_content})


def evidence_record(document: Document) -> dict:
    """JSON-ready evidence for citations and evaluation traces."""
    doc = prepare_evidence([document])[0]
    keys = ("document_id", "chunk_id", "source", "filename", "page", "page_number", "start_index")
    return {**{key: doc.metadata[key] for key in keys},
            "page_content": doc.page_content, "metadata": dict(doc.metadata)}


def resolve_citations(answer: str, documents: list[Document]) -> dict:
    """Resolve inline [chunk_...] references only against the retrieved passages.

    Unknown IDs are reported and never added to the citations list. Membership
    validation does not prove that a passage semantically supports a claim.
    """
    if not isinstance(answer, str):
        raise TypeError("Citation resolution requires a text answer")
    available = {record["chunk_id"]: record for record in map(evidence_record, documents)}
    referenced = list(dict.fromkeys(CITATION_PATTERN.findall(answer)))
    invalid = [chunk_id for chunk_id in referenced if chunk_id not in available]
    missing = not referenced and answer.strip().casefold() != ABSTENTION_ANSWER.casefold()
    return {
        "citations": [available[chunk_id] for chunk_id in referenced if chunk_id in available],
        "citation_validation": {
            "valid": not invalid and not missing,
            "invalid_chunk_ids": invalid,
            "missing_citations": missing,
        },
    }
