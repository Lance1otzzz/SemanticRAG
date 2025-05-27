from chonkie import Chonkie, DefaultChunker
from chonkie.types import Document  # Assuming Chonkie uses a Document type
from typing import List, Optional, Dict, Any

# You might need to import specific chunkers if not using DefaultChunker directly
# from chonkie.chunkers import SomeSpecificChunker

class ChonkieTextChunker:
    def __init__(self, chunker_name: str = "DefaultChunker", chunker_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ChonkieTextChunker.

        Args:
            chunker_name (str): The name of the Chonkie chunker to use. 
                                  Currently, only "DefaultChunker" is implemented as an example.
                                  This can be expanded to select other Chonkie chunkers.
            chunker_config (Optional[Dict[str, Any]]): Configuration dictionary for the chunker.
                                                       Refer to Chonkie documentation for specific chunker parameters.
        """
        if chunker_config is None:
            chunker_config = {}

        # Placeholder for selecting different chunkers
        # For now, we'll use DefaultChunker or allow passing a pre-configured Chonkie instance
        if chunker_name == "DefaultChunker":
            # Example: DefaultChunker might take parameters like chunk_size, chunk_overlap
            # These would be passed in chunker_config, e.g., {"chunk_size": 1000, "chunk_overlap": 200}
            # Adjust based on actual DefaultChunker or other Chonkie chunker parameters
            self.chunker = Chonkie(chunker=DefaultChunker(**chunker_config))
        # Example of how to add another chunker:
        # elif chunker_name == "RecursiveCharacterTextSplitter": # Assuming Chonkie has this
        #     from chonkie.text_splitters import RecursiveCharacterTextSplitter # Fictional import
        #     self.chunker = Chonkie(chunker=RecursiveCharacterTextSplitter(**chunker_config))
        else:
            raise ValueError(f"Unsupported chunker_name: {chunker_name}. Please use 'DefaultChunker' or extend this class.")

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Chunks the input text using the configured Chonkie chunker.

        Args:
            text (str): The text to be chunked.
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with the created Document.

        Returns:
            List[Document]: A list of Chonkie Document objects representing the chunks.
        """
        if not text:
            return []
            
        # Chonkie's process method might take Document directly or text.
        # Adjust based on Chonkie's API. Assuming it can take text and metadata for a Document.
        # If Chonkie expects a Document object as input to process, create one first.
        # For example:
        # initial_document = Document(content=text, metadata=metadata if metadata else {})
        # chunks = self.chunker.process([initial_document]) 
        
        # Simpler approach if Chonkie can directly chunk text string:
        # This is a common pattern, but Chonkie's specific API might differ.
        # The following is a conceptual representation.
        # We might need to create a Document first, then chunk it.
        
        # According to Chonkie docs, it seems to operate on a list of Documents.
        # So we create one initial document.
        doc = Document(content=text, metadata=metadata if metadata else {})
        
        # The .chunk() method is available on Chonkie instances.
        # It returns a list of Document objects.
        chunked_documents = self.chunker.chunk([doc]) # Pass a list of documents
        
        return chunked_documents

# Example Usage (for testing within this file, can be removed or moved to main.py later):
if __name__ == '__main__':
    sample_text = (
        "This is the first sentence. This is the second sentence, which is a bit longer. "
        "Here comes the third sentence. The fourth sentence follows. And finally, the fifth sentence."
    )
    
    # Using DefaultChunker with its default settings
    try:
        print("Testing with DefaultChunker (default config)...")
        default_chunker_config = {"chunk_size": 50, "chunk_overlap": 10} # Example config
        chunker_instance = ChonkieTextChunker(chunker_name="DefaultChunker", chunker_config=default_chunker_config)
        chunks = chunker_instance.chunk_text(sample_text, metadata={"source": "test_document"})
        
        if chunks:
            for i, chunk in enumerate(chunks):
                print(f"--- Chunk {i+1} ---")
                print(f"Content: {chunk.content}")
                print(f"Metadata: {chunk.metadata}")
                print(f"Tokens: {chunk.tokens}") # If Chonkie populates this
        else:
            print("No chunks were produced.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        print("Please ensure Chonkie is installed and the API usage is correct.")
        print("Refer to Chonkie documentation for DefaultChunker parameters and usage.")
