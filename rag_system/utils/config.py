# API Key Management and Configuration

# Placeholder for OpenAI API Key
# Replace "YOUR_OPENAI_API_KEY_HERE" with your actual OpenAI API key if you plan to use OpenAI embeddings.
# Example: OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

# Placeholder for API keys potentially used by ChromaDB's internal embedding functions
# For example, if you use ChromaDB's OpenAIEmbeddingFunction directly via ChromaDBManager.
# Replace "YOUR_API_KEY_IF_NEEDED_BY_CHROMA_EMBED_FN" with the relevant key.
CHROMA_OPENAI_API_KEY = "YOUR_API_KEY_IF_NEEDED_BY_CHROMA_EMBED_FN" 

# Other configurations can be added here, for example:
# DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CHROMA_DB_PATH = "./persistent_chroma_db"
# DEFAULT_COLLECTION_NAME = "my_default_collection"

print("config.py loaded. Ensure API keys are configured if using services like OpenAI.")
