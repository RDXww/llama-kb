"""llama-kb: local bilingual RAG for llama.cpp and OpenAI-compatible APIs.

Modules:
    config      configuration and paths
    text        loading, chunking, tokenization, language detection
    catalog     document groups, aliases, versions
    router      group-aware query routing
    embedder    multilingual embedding model
    lexical     SQLite FTS5 keyword index
    bm25        pure-Python BM25 used by the tests and legacy path
    store       Chroma persistence, FTS5 sync, incremental manifest
    retriever   hybrid dense + lexical retrieval and evidence gating
    prompt      bilingual prompting, citations, source payloads
    server      OpenAI-compatible RAG proxy
    chat        terminal chat client
    cli         command-line interface
    importers   optional document importers (DevDocs)
"""

__version__ = "0.1.0"
