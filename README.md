# SemanticRAG

This repository contains a lightweight prototype of a Retrieval-Augmented Generation (RAG)
system alongside an MCP-based server.  The RAG components live in `rag_system/`
while the server entry point is located under `mcp_server/`.

The RAG system is structured into independent modules:

- **chunking** – breaks raw text into semantically meaningful chunks and records
  structural metadata.
- **embedding** – generates vector representations for text and associates them
  with the metadata.
- **vector_store** – wraps a ChromaDB collection for storing and querying
  embeddings.

The server in `mcp_server/` exposes a small API built with FastMCP.  **Do not
modify the server files when extending the RAG modules.**

## Development setup

1. Create a Python environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r rag_system/requirements.txt
   ```

2. Run the unit tests to verify the basic functionality:

   ```bash
   python -m unittest discover tests
   ```

3. To start the MCP server (requires the `mcp` package):

   ```bash
   mcp dev mcp_server/semantic_rag_server.py
   ```

## Repository Layout

```
README.md
mcp_server/              # FastMCP entry point
rag_system/
    chunking/
    embedding/
    vector_store/
    utils/
```

The repository is intentionally minimal; see `DEVLOG.md` for a short summary of
implemented changes.
