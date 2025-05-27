import chromadb
from chromadb.utils import embedding_functions # For potential future use with ChromaDB's own embedding functions
from typing import List, Dict, Optional, Any, Union
import uuid # For generating unique IDs for documents

# Placeholder for where the user might specify their API key for an embedding function
# if they choose to use one provided by ChromaDB that requires it (e.g., OpenAIEmbeddingFunction)
CHROMA_EMBEDDING_FUNCTION_API_KEY_PLACEHOLDER = "YOUR_API_KEY_IF_NEEDED_BY_CHROMA_EMBED_FN"

class ChromaDBManager:
    def __init__(self, path: str = "./chroma_data", collection_name: str = "my_collection",
                 embedding_model_name: Optional[str] = "all-MiniLM-L6-v2", # Used for metadata, if needed
                 embedding_function_api_key: Optional[str] = None,
                 embedding_function_name: Optional[str] = None): # Name of ChromaDB built-in EF
        """
        Initializes the ChromaDBManager.

        Args:
            path (str): Path to the directory where ChromaDB data will be persisted.
            collection_name (str): Name of the collection to create or load.
            embedding_model_name (Optional[str]): Name of the embedding model being used externally.
                                                 Stored as metadata in the collection.
            embedding_function_api_key (Optional[str]): API key if using a ChromaDB built-in EF that needs one.
            embedding_function_name (Optional[str]): The name of a ChromaDB built-in embedding function to use
                                                     (e.g., 'OpenAIEmbeddingFunction'). If None, assumes embeddings
                                                     are generated externally and passed to add_documents.
        """
        try:
            self.client = chromadb.PersistentClient(path=path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB PersistentClient at path '{path}'. Error: {e}")

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Prepare embedding function if specified
        self.embedding_function = None
        if embedding_function_name:
            api_key_to_use = embedding_function_api_key if embedding_function_api_key else CHROMA_EMBEDDING_FUNCTION_API_KEY_PLACEHOLDER
            if embedding_function_name.lower() == 'openaiembeddingfunction':
                if api_key_to_use == CHROMA_EMBEDDING_FUNCTION_API_KEY_PLACEHOLDER or not api_key_to_use:
                    print(f"Warning: OpenAI API key not provided for ChromaDB's OpenAIEmbeddingFunction. It may not work.")
                # Requires chromadb[openai] to be installed
                try:
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key_to_use,
                        model_name=embedding_model_name if embedding_model_name else "text-embedding-ada-002" # Default model for OpenAI
                    )
                except Exception as e:
                    print(f"Failed to initialize OpenAIEmbeddingFunction for ChromaDB: {e}")
                    self.embedding_function = None # Fallback
            elif embedding_function_name.lower() == 'sentencetransformerembeddingfunction':
                 # Requires chromadb[sentence-transformers] to be installed
                try:
                    self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=embedding_model_name if embedding_model_name else "all-MiniLM-L6-v2"
                    )
                except Exception as e:
                    print(f"Failed to initialize SentenceTransformerEmbeddingFunction for ChromaDB: {e}")
                    self.embedding_function = None # Fallback
            else:
                print(f"Warning: Specified embedding_function_name '{embedding_function_name}' is not directly supported by this class's examples. "
                      "ChromaDB will use its default if not set, or you can pass a custom one.")

        # Metadata for collection creation, including HNSW settings
        # User requirement: Use HNSW. This is specified in collection metadata.
        # Common spaces: 'l2' (Euclidean), 'ip' (inner product), 'cosine'
        # ChromaDB defaults to 'l2' if not specified. 'cosine' is often good for sentence embeddings.
        collection_metadata = {"hnsw:space": "cosine"} # Using cosine similarity
        if self.embedding_model_name:
            collection_metadata["embedding_model"] = self.embedding_model_name
        
        # Add a placeholder for future/other index types or configurations
        collection_metadata["notes"] = "Designed for extensibility with other potential metadata or index configurations."

        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=collection_metadata,
                embedding_function=self.embedding_function # Pass EF if defined
            )
            # print(f"Successfully got or created collection '{self.collection_name}' with HNSW (cosine space).")
            # print(f"Collection metadata: {self.collection.metadata}") # Verify metadata
        except Exception as e:
            # This can happen if the collection exists with a different embedding function or metadata (like hnsw:space)
            # Or if there's an issue with the database itself.
            print(f"Error getting or creating collection '{self.collection_name}': {e}")
            print("Attempting to get collection without metadata to check if it exists with conflicting settings...")
            try:
                self.collection = self.client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
                print(f"Successfully retrieved existing collection '{self.collection_name}'.")
                print(f"NOTE: The existing collection's HNSW settings or other metadata might differ from the desired ones if it was created previously with different parameters.")
            except Exception as e_get:
                 raise RuntimeError(f"Failed to get or create collection '{self.collection_name}'. Initial error: {e}. Get error: {e_get}")


    def add_documents(self, texts: List[str], embeddings: Optional[List[List[float]]] = None, 
                      metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        """
        Adds documents (texts and their embeddings) to the ChromaDB collection.
        If an embedding_function was set on the collection, `embeddings` can be None.
        Otherwise, `embeddings` are required.

        Args:
            texts (List[str]): The list of text documents.
            embeddings (Optional[List[List[float]]]): A list of embeddings corresponding to the texts. 
                                                     Required if no embedding_function is set on the collection.
            metadatas (Optional[List[Dict[str, Any]]]): A list of metadata dictionaries for each document.
                                                       Designed to be extensible.
            ids (Optional[List[str]]): A list of unique IDs for each document. If None, UUIDs are generated.
        """
        if not texts:
            print("No texts provided to add_documents.")
            return

        if self.embedding_function is None and embeddings is None:
            raise ValueError("Embeddings must be provided if the collection does not have an embedding function.")
        
        if embeddings is not None and len(texts) != len(embeddings):
            raise ValueError("The number of texts and embeddings must be the same.")

        if metadatas is not None and len(texts) != len(metadatas):
            raise ValueError("The number of texts and metadatas must be the same.")
        
        if ids is not None and len(texts) != len(ids):
            raise ValueError("The number of texts and ids must be the same.")

        processed_ids = ids if ids else [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadata is a list of dicts, even if some are None
        processed_metadatas = []
        if metadatas:
            processed_metadatas = [m if m is not None else {} for m in metadatas]
        else:
            processed_metadatas = [{} for _ in texts]

        # Add source text to metadata for easier inspection if not already there
        for i, meta in enumerate(processed_metadatas):
            if 'original_text' not in meta: # Or some other preferred key
                meta['original_text'] = texts[i][:500] # Store a snippet or full text

        try:
            if embeddings: # If providing embeddings directly
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts, # Chroma also stores the document text itself
                    metadatas=processed_metadatas,
                    ids=processed_ids
                )
            else: # If relying on collection's embedding function
                self.collection.add(
                    documents=texts,
                    metadatas=processed_metadatas,
                    ids=processed_ids
                )
            # print(f"Successfully added {len(texts)} documents to collection '{self.collection_name}'.")
        except Exception as e:
            # Consider batching for very large additions if errors occur
            print(f"Error adding documents to ChromaDB: {e}")
            # You might want to implement batching or more robust error handling here
            # For example, ChromaDB client might have limits on batch size.
            # Example: if "DefaultDimensionality είναι different from CollectionDimensionality" error,
            # it means embedding dimension mismatch if EF is used.

    def get_collection_info(self) -> Dict[str, Any]:
        """Returns information about the collection."""
        return {
            "name": self.collection.name,
            "id": self.collection.id,
            "count": self.collection.count(),
            "metadata": self.collection.metadata, # Includes HNSW settings
        }

    def query_collection(self, query_embeddings: Optional[List[List[float]]] = None, 
                         query_texts: Optional[List[str]] = None,
                         n_results: int = 5, 
                         where_filter: Optional[Dict[str, Any]] = None,
                         include: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Queries the collection for documents similar to the query embeddings or texts.
        If the collection has an embedding function, query_texts can be used directly.
        Otherwise, query_embeddings must be provided.

        Args:
            query_embeddings (Optional[List[List[float]]]): A list of query embeddings.
            query_texts (Optional[List[str]]): A list of query texts (if collection has EF).
            n_results (int): The number of results to return.
            where_filter (Optional[Dict[str, Any]]): A metadata filter. 
                Example: {"source": "doc_A"} or {"custom_field": {"$eq": "value1"}}
            include (Optional[List[str]]): A list of fields to include in the results.
                Default: ["metadatas", "documents", "distances"]. Can also include "embeddings".

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the query results (ids, documents,
                                     metadatas, distances, etc.), or None if querying fails.
                                     The structure is like:
                                     {
                                         'ids': [[id1, id2]], 
                                         'documents': [[doc1, doc2]],
                                         'metadatas': [[meta1, meta2]],
                                         'distances': [[dist1, dist2]]
                                     }
                                     Each inner list corresponds to a query in query_embeddings/query_texts.
        """
        if self.collection is None:
            print("Error: Collection is not initialized.")
            return None

        if not query_embeddings and not query_texts:
            raise ValueError("Either query_embeddings or query_texts must be provided.")

        if query_embeddings is None and query_texts and self.embedding_function is None:
            raise ValueError("query_embeddings must be provided if the collection does not have an embedding function.")

        if include is None:
            include = ["metadatas", "documents", "distances"] # Common useful fields

        try:
            if query_embeddings:
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where_filter,
                    include=include
                )
            elif query_texts: # This implies self.embedding_function is not None
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where_filter,
                    include=include
                )
            else:
                # Should not happen due to checks above, but as a fallback
                print("Error: Invalid query parameters.")
                return None
            
            return results
        except Exception as e:
            print(f"Error during collection query: {e}")
            return None

# Example Usage (for testing, can be moved to main.py later):
if __name__ == '__main__':
    print("--- Testing ChromaDBManager ---")
    try:
        # Test 1: Initialize with external embeddings (typical case for this project)
        print("\n--- Test 1: External Embeddings ---")
        chroma_manager_ext = ChromaDBManager(path="./chroma_data_test_ext", collection_name="test_collection_ext", embedding_model_name="test_model_v1")
        
        sample_texts_ext = ["doc1: hello world", "doc2: foo bar"]
        # These dimensions should match what 'test_model_v1' would produce
        sample_embeddings_ext = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] 
        sample_metadatas_ext = [{"source": "doc_A"}, {"source": "doc_B", "custom_field": "value1"}]
        sample_ids_ext = ["id1_ext", "id2_ext"]

        chroma_manager_ext.add_documents(
            texts=sample_texts_ext, 
            embeddings=sample_embeddings_ext, 
            metadatas=sample_metadatas_ext, 
            ids=sample_ids_ext
        )
        print("Documents added (external embeddings).")
        info_ext = chroma_manager_ext.get_collection_info()
        print(f"Collection Info (external): {info_ext}")
        retrieved = chroma_manager_ext.collection.get(ids=["id1_ext"])
        print(f"Retrieved id1_ext: {retrieved['documents']}, {retrieved['metadatas']}")

        # Test 2: Using ChromaDB's SentenceTransformer Embedding Function
        # This requires `pip install chromadb[sentence-transformers]`
        print("\n--- Test 2: ChromaDB SentenceTransformer EF ---")
        # Ensure the model name is valid for sentence-transformers
        st_model_for_chroma = "all-MiniLM-L6-v2" 
        try:
            chroma_manager_st = ChromaDBManager(
                path="./chroma_data_test_st_ef", 
                collection_name="test_collection_st_ef",
                embedding_function_name='SentenceTransformerEmbeddingFunction',
                embedding_model_name=st_model_for_chroma 
            )
            sample_texts_st = ["Chroma uses ST for this text.", "Another text for ST."]
            sample_metadatas_st = [{"source": "st_A"}, {"source": "st_B"}]
            # Embeddings are NOT provided, Chroma will generate them
            chroma_manager_st.add_documents(texts=sample_texts_st, metadatas=sample_metadatas_st)
            print("Documents added (Chroma ST EF).")
            info_st = chroma_manager_st.get_collection_info()
            print(f"Collection Info (Chroma ST EF): {info_st}")
            if info_st["count"] > 0:
                 # Verify dimension from actual embedding if possible or check metadata
                peek_result = chroma_manager_st.collection.peek(limit=1)
                if peek_result and peek_result.get('embeddings'):
                    print(f"Dimension from ST EF: {len(peek_result['embeddings'][0])}")
                else:
                    print("Could not peek embeddings for ST EF.")
        except Exception as e_st:
            print(f"Error during ChromaDB ST EF test: {e_st}")
            print("This test might fail if 'chromadb[sentence-transformers]' is not installed or model download fails.")

        # Clean up test directories (optional)
        # import shutil
        # shutil.rmtree("./chroma_data_test_ext", ignore_errors=True)
        # shutil.rmtree("./chroma_data_test_st_ef", ignore_errors=True)
        # print("\nCleaned up test directories.")

        # Test 3: Querying the collection (external embeddings example)
        print("\n--- Test 3: Querying Collection (External Embeddings) ---")
        if chroma_manager_ext and chroma_manager_ext.collection.count() > 0:
            # Query with an embedding similar to "doc1: hello world" -> [1.0, 2.0, 3.0]
            # For a real scenario, this query_embedding would come from the TextEmbedder
            query_embedding_ext = [[1.1, 2.1, 3.1]] # Slightly different to test similarity
            
            retrieved_docs_ext = chroma_manager_ext.query_collection(
                query_embeddings=query_embedding_ext,
                n_results=1,
                include=["metadatas", "documents", "distances"]
            )
            if retrieved_docs_ext and retrieved_docs_ext.get('documents'):
                print(f"Query results (external):")
                print(f"  IDs: {retrieved_docs_ext['ids']}")
                print(f"  Documents: {retrieved_docs_ext['documents']}")
                print(f"  Metadatas: {retrieved_docs_ext['metadatas']}")
                print(f"  Distances: {retrieved_docs_ext['distances']}")
            else:
                print("No documents found or error in query (external).")
        else:
            print("Skipping query test for external embeddings as collection is empty or manager not init.")

        # Test 4: Querying the collection (Chroma ST EF example)
        print("\n--- Test 4: Querying Collection (Chroma ST EF) ---")
        if 'chroma_manager_st' in locals() and chroma_manager_st and chroma_manager_st.collection.count() > 0:
            query_text_st = "What is Chroma ST doing?" # A query text
            
            retrieved_docs_st = chroma_manager_st.query_collection(
                query_texts=[query_text_st], # Use query_texts because this collection has an EF
                n_results=1,
                include=["metadatas", "documents", "distances"]
            )
            if retrieved_docs_st and retrieved_docs_st.get('documents'):
                print(f"Query results (Chroma ST EF) for '{query_text_st}':")
                print(f"  IDs: {retrieved_docs_st['ids']}")
                print(f"  Documents: {retrieved_docs_st['documents']}")
                print(f"  Metadatas: {retrieved_docs_st['metadatas']}")
                print(f"  Distances: {retrieved_docs_st['distances']}")
            else:
                print(f"No documents found or error in query for '{query_text_st}' (Chroma ST EF).")
        else:
            print("Skipping query test for Chroma ST EF as collection is empty or manager not init.")

    except Exception as e:
        print(f"An error occurred during ChromaDBManager testing: {e}")
