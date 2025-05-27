from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np

# Import config settings if you have API keys there
# from rag_system.utils import config 

# Define a placeholder for API keys directly in the code or fetch from config.py
# This is where the user should insert their key if using a service like OpenAI.
OPENAI_API_KEY_PLACEHOLDER = "YOUR_OPENAI_API_KEY_HERE" 

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_service: str = "sentence-transformers", api_key: Optional[str] = None):
        """
        Initializes the TextEmbedder.

        Args:
            model_name (str): The name of the embedding model to use.
                              For "sentence-transformers", this is a model like 'all-MiniLM-L6-v2'.
                              For "openai", this could be 'text-embedding-ada-002'.
            embedding_service (str): The service to use for embeddings. 
                                     Supported: "sentence-transformers", "openai".
            api_key (Optional[str]): The API key for the embedding service (e.g., OpenAI).
                                     If None, it will try to use OPENAI_API_KEY_PLACEHOLDER or an environment variable.
        """
        self.model_name = model_name
        self.embedding_service = embedding_service.lower()
        self.client = None

        if self.embedding_service == "sentence-transformers":
            try:
                self.client = SentenceTransformer(model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}'. Ensure it's a valid model and library is installed correctly. Error: {e}")
        elif self.embedding_service == "openai":
            try:
                from openai import OpenAI # Lazy import
            except ImportError:
                raise ImportError("OpenAI library not found. Please install it using 'pip install openai'")
            
            self.api_key = api_key if api_key else OPENAI_API_KEY_PLACEHOLDER
            if self.api_key == "YOUR_OPENAI_API_KEY_HERE" or not self.api_key:
                print("Warning: OpenAI API key not provided or is using the placeholder. OpenAI embedding will not work.")
                self.client = None # Ensure client is None if API key is missing
            else:
                self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported embedding_service: {embedding_service}. Choose 'sentence-transformers' or 'openai'.")

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Embeds a list of text strings.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            Optional[List[List[float]]]: A list of embeddings (list of floats for each text), 
                                         or None if embedding fails (e.g., API key issue).
        """
        if not texts:
            return []

        if self.embedding_service == "sentence-transformers":
            if self.client:
                embeddings = self.client.encode(texts, convert_to_numpy=True)
                return embeddings.tolist() # Convert numpy arrays to lists of floats
            else:
                print("Error: SentenceTransformer client not initialized.")
                return None
        elif self.embedding_service == "openai":
            if self.client and self.api_key != "YOUR_OPENAI_API_KEY_HERE" and self.api_key:
                try:
                    response = self.client.embeddings.create(
                        input=texts,
                        model=self.model_name # e.g., "text-embedding-ada-002"
                    )
                    return [item.embedding for item in response.data]
                except Exception as e:
                    print(f"Error during OpenAI embedding: {e}")
                    return None
            else:
                print("Error: OpenAI client not initialized or API key missing. Cannot embed.")
                return None
        return None

    def get_embedding_dimension(self) -> Optional[int]:
        """
        Returns the dimension of the embeddings produced by the model.
        """
        if self.embedding_service == "sentence-transformers":
            if self.client:
                return self.client.get_sentence_embedding_dimension()
            return None 
        elif self.embedding_service == "openai":
            if self.model_name == "text-embedding-ada-002":
                return 1536
            print(f"Warning: Dimension for OpenAI model '{self.model_name}' is not explicitly defined here. Returning a common default or None.")
            # This is a common dimension for OpenAI models, but might not be accurate for all.
            # Ideally, fetch this from OpenAI API or have a mapping for known models.
            return 1536 # Placeholder, adjust if necessary for other OpenAI models
        return None


# Example Usage (for testing, can be moved to main.py later):
if __name__ == '__main__':
    sample_texts = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

    print("--- Testing with Sentence-Transformers (all-MiniLM-L6-v2) ---")
    try:
        st_embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", embedding_service="sentence-transformers")
        st_embeddings = st_embedder.embed_texts(sample_texts)
        if st_embeddings:
            print(f"Successfully got {len(st_embeddings)} embeddings.")
            print(f"Dimension: {st_embedder.get_embedding_dimension()}")
            # print(f"First embedding: {st_embeddings[0][:5]}...") # Print first 5 dims of first embedding
        else:
            print("Failed to get Sentence-Transformer embeddings.")
    except Exception as e:
        print(f"Error testing Sentence-Transformers: {e}")

    print("\n--- Testing with OpenAI (Placeholder - will not work without a key) ---")
    try:
        # This test will likely print a warning or fail if the API key is the placeholder
        openai_embedder = TextEmbedder(model_name="text-embedding-ada-002", embedding_service="openai", api_key="YOUR_OPENAI_API_KEY_HERE") 
        
        if openai_embedder.client: # Check if client was initialized (it shouldn't be with placeholder key)
            openai_embeddings = openai_embedder.embed_texts(["Test OpenAI text"])
            if openai_embeddings:
                print(f"Successfully got {len(openai_embeddings)} OpenAI embeddings.")
                print(f"Dimension: {openai_embedder.get_embedding_dimension()}")
            else:
                print("Failed to get OpenAI embeddings (API key might be missing or invalid).")
        else:
            print("OpenAI client not initialized, likely due to missing API key.")
            
    except Exception as e:
        # This might catch the ValueError from __init__ if the service name was wrong,
        # or other issues during initialization or embedding.
        print(f"Error testing OpenAI: {e}")
