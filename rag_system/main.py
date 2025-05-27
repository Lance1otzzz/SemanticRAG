import os
import shutil # For cleaning up Chroma data if needed

# Import the necessary classes from our modules
from chunking.chonkie_chunker import ChonkieTextChunker, Document as ChonkieDocument # Assuming Document is the type from Chonkie
from embedding.embedder import TextEmbedder
from vector_store.chroma_db import ChromaDBManager
from typing import List # Required for type hinting

# Configuration (can be moved to utils.config.py later)
SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. This sentence is about a speedy fox and a relaxed dog. "
    "Chonkie is a library for text chunking, which can break down large texts into smaller pieces. "
    "Embeddings are numerical representations of text, often used in semantic search. "
    "ChromaDB is a vector database that stores these embeddings for efficient retrieval. "
    "The goal of a RAG system is to retrieve relevant information to augment LLM responses."
)
CHONKIE_CONFIG = {"chunk_size": 50, "chunk_overlap": 10} # Example, adjust as per Chonkie's DefaultChunker actual params
EMBEDDER_SERVICE = "sentence-transformers" # "openai" or "sentence-transformers"
EMBEDDER_MODEL_ST = "all-MiniLM-L6-v2" # For sentence-transformers
EMBEDDER_MODEL_OPENAI = "text-embedding-ada-002" # For OpenAI
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE" # Replace if using OpenAI

CHROMA_PATH = "./rag_chroma_data_main"
CHROMA_COLLECTION_NAME = "main_demo_collection"

def cleanup_chroma_data(path: str = CHROMA_PATH):
    """Removes the ChromaDB data directory for a clean run."""
    if os.path.exists(path):
        print(f"Cleaning up old ChromaDB data at: {path}")
        shutil.rmtree(path)

def main():
    print("--- Starting RAG System Demo ---")

    # 0. Optional: Cleanup previous Chroma data for a fresh start
    cleanup_chroma_data()

    # 1. Initialize Chonkie Text Chunker
    print("\n--- 1. Text Chunking ---")
    try:
        # Note: Ensure Chonkie's DefaultChunker parameters are correctly named if different from chunk_size/overlap
        chunker = ChonkieTextChunker(chunker_name="DefaultChunker", chunker_config=CHONKIE_CONFIG)
        # The metadata for the initial document can be simple for this example
        text_chunks_docs: List[ChonkieDocument] = chunker.chunk_text(SAMPLE_TEXT, metadata={"source": "sample_document_main"})
        
        if not text_chunks_docs:
            print("No chunks produced by Chonkie. Exiting.")
            return

        text_chunks_content: List[str] = [doc.content for doc in text_chunks_docs]
        print(f"Successfully chunked text into {len(text_chunks_content)} chunks.")
        for i, chunk_doc in enumerate(text_chunks_docs):
            print(f"  Chunk {i+1}: '{chunk_doc.content}' (Metadata: {chunk_doc.metadata})")
    except Exception as e:
        print(f"Error during text chunking: {e}")
        return

    # 2. Initialize Text Embedder
    print("\n--- 2. Text Embedding ---")
    embedder_model = EMBEDDER_MODEL_ST if EMBEDDER_SERVICE == "sentence-transformers" else EMBEDDER_MODEL_OPENAI
    api_key_to_use = OPENAI_API_KEY if EMBEDDER_SERVICE == "openai" else None
    
    if EMBEDDER_SERVICE == "openai" and (api_key_to_use == "YOUR_OPENAI_API_KEY_HERE" or not api_key_to_use):
        print("OpenAI API key is a placeholder. Please replace it in `main.py` or `utils/config.py` to use OpenAI embeddings.")
        print("Skipping embedding and further steps that depend on it.")
        # Fallback or exit
        # Alternatively, you could switch to sentence-transformers here if OpenAI key is missing
        # embedder_service_to_use = "sentence-transformers"
        # embedder_model_to_use = EMBEDDER_MODEL_ST
        # print("Falling back to sentence-transformers due to missing OpenAI key.")
        # embedder = TextEmbedder(model_name=embedder_model_to_use, embedding_service=embedder_service_to_use)
        return # For this demo, we'll just stop if the selected service can't run.
        
    try:
        embedder = TextEmbedder(model_name=embedder_model, embedding_service=EMBEDDER_SERVICE, api_key=api_key_to_use)
        chunk_embeddings = embedder.embed_texts(text_chunks_content)

        if not chunk_embeddings or len(chunk_embeddings) == 0:
            print("Failed to generate embeddings. Exiting.")
            return
        print(f"Successfully generated {len(chunk_embeddings)} embeddings.")
        print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    except Exception as e:
        print(f"Error during text embedding: {e}")
        return

    # 3. Initialize ChromaDB Manager and Add Documents
    print("\n--- 3. Indexing in ChromaDB ---")
    try:
        chroma_manager = ChromaDBManager(
            path=CHROMA_PATH, 
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_model_name=embedder_model # Storing the model name as metadata
        )
        
        # Prepare metadatas for ChromaDB - Chonkie might already have some
        # For this example, let's use the metadata from Chonkie documents
        # and add the chunk index.
        doc_metadatas = []
        for i, chunk_doc in enumerate(text_chunks_docs):
            meta = chunk_doc.metadata.copy() if chunk_doc.metadata else {}
            meta["chunk_index"] = i
            # meta["original_text_preview"] = chunk_doc.content[:100] # Already added by ChromaDBManager
            doc_metadatas.append(meta)

        # Generate unique IDs for each chunk
        doc_ids = [f"chunk_{i}_{abs(hash(chunk_doc.content[:50]))}" for i, chunk_doc in enumerate(text_chunks_docs)]

        chroma_manager.add_documents(
            texts=text_chunks_content,
            embeddings=chunk_embeddings,
            metadatas=doc_metadatas,
            ids=doc_ids
        )
        print(f"Successfully added {len(text_chunks_content)} documents to ChromaDB collection '{CHROMA_COLLECTION_NAME}'.")
        collection_info = chroma_manager.get_collection_info()
        print(f"Collection info: Count={collection_info.get('count')}, Metadata={collection_info.get('metadata')}")

    except Exception as e:
        print(f"Error during ChromaDB operations: {e}")
        return

    # 4. Perform a Retrieval Query
    print("\n--- 4. Retrieval Query ---")
    query_text = "What is ChromaDB?"
    # query_text = "Tell me about the fox."
    print(f"Querying with: '{query_text}'")

    try:
        # Embed the query text using the same embedder
        query_embedding = embedder.embed_texts([query_text])
        if not query_embedding:
            print("Failed to embed query text. Exiting.")
            return

        retrieved_results = chroma_manager.query_collection(
            query_embeddings=query_embedding,
            n_results=2 # Get top 2 results
        )

        if retrieved_results and retrieved_results.get('documents'):
            print("Retrieved documents:")
            for i in range(len(retrieved_results['ids'][0])): # Results are nested per query
                doc_id = retrieved_results['ids'][0][i]
                doc_content = retrieved_results['documents'][0][i]
                doc_distance = retrieved_results['distances'][0][i]
                doc_metadata = retrieved_results['metadatas'][0][i]
                print(f"  Rank {i+1}:")
                print(f"    ID: {doc_id}")
                print(f"    Distance: {doc_distance:.4f}")
                print(f"    Content: '{doc_content}'")
                print(f"    Metadata: {doc_metadata}")
        else:
            print("No results found or error during query.")
            if retrieved_results: # Print if structure exists but no docs
                 print(f"Raw results structure: {retrieved_results}")


    except Exception as e:
        print(f"Error during retrieval query: {e}")
        return
        
    print("\n--- RAG System Demo Finished ---")

if __name__ == "__main__":
    main()
