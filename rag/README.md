# Source evidence and citations

`prepare_resources()` now gives every chunk stable source metadata before
indexing. `split_documents()` records character offsets with `add_start_index`.

- `source`: PDF path relative to the loaded collection root, including subfolders.
- `filename`: source basename for display.
- `document_id`: deterministic hash of the normalized source location.
- `chunk_id` / `Document.id`: deterministic hash of document ID, page, character
  offset, and exact chunk text. Reindexing identical inputs produces identical IDs;
  changing source, page, position, or text changes the chunk ID.
- `page`: original **zero-based** PDF page index.
- `page_number`: **one-based** PDF page position for display, not a printed page
  label (which may use Roman numerals or restart within a document).
- `start_index`: zero-based character offset within the loader's page text.

Document IDs identify logical source locations, not file revisions. Different
collection roots with the same relative file paths are separate collections;
namespace their source paths before combining them into one index. Custom or
legacy documents without provenance are explicitly labeled `unknown`, with
`page_number=None`; no page is guessed. A custom document's `page`, when present,
must follow the same zero-based convention.

Hybrid retrieval fuses the same chunk across dense/BM25 results using `chunk_id`.
Identical text in different documents, pages, or offsets remains separate evidence.

The default generation prompt includes source, display page, and `[chunk_...]`
headers for every passage. Retrieved text is reference data in the human message;
the system message tells the model not to follow instructions found in passages.
The expected answer includes inline references, for example:

```text
TCP provides reliable delivery [chunk_<actual retrieved ID>].
```

Use one pair of brackets per ID. The pipeline still returns `answer` as text and
`context` as Documents, and adds:

```python
{
    "citations": [
        # Resolved sources in first-citation order, with duplicate IDs removed.
        # Each contains chunk_id, document_id, source, filename, page, page_number,
        # start_index, page_content, and original/enriched metadata.
    ],
    "citation_validation": {
        "valid": True,
        "invalid_chunk_ids": [],
        "missing_citations": False,
    },
}
```

Unknown IDs, even real chunks outside the retrieved context, are flagged and never
turned into resolved source records. An answer without references is flagged
unless it is the exact supported abstention: `I don't know based on the provided
documents.` The original answer is retained for debugging/evaluation, including
invalid inline references; consumers must check `citation_validation` before
presenting an answer as verified and create links only from resolved `citations`.
Membership validation does **not** establish that a passage supports the claim,
or check citation coverage sentence by sentence. Groundedness evaluation remains
necessary. Custom prompts/document formatters must follow the same reference
convention to obtain valid citation reports. `assemble()` expects a text answer;
standalone `build_generation()` still supports custom output parsers.

The evaluation adapter preserves the same headers in the judge's text context
and adds `source_documents` with structured records. It also forwards `citations`
and `citation_validation` from the pipeline, so LangSmith traces retain the evidence.

Existing saved FAISS indexes are not rewritten automatically. Rebuild using
`prepare_resources(save_path=...)` to index canonical relative sources and offsets.
Legacy retrieved chunks receive deterministic fallback IDs from their existing
metadata, but cannot recover character offsets that were never saved.

Run the offline tests from the repository root:

```bash
LANGSMITH_TRACING=false venv/bin/python -B -m unittest discover -s tests -v
```

The tests inject fake generation, embedding/retrieval resources, and reranking
models, and do not download models or call hosted APIs.
