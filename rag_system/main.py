import os
import shutil # For cleaning up Chroma data if needed

# Import the necessary classes from our modules
from chunking.chonkie_chunker import ChonkieTextChunker, ChonkieDocument
from chunking.chapter_aware_chunker import ChapterAwareChunker, convert_to_chonkie_documents
from chunking.metadata_extractor import MetadataExtractor
from chunking.llm_chunker import LLMChunker, Chunk
from embedding.embedder import TextEmbedder
from rag_system.chunking.llm_chunker import GeminiClient
from vector_store.chroma_db import ChromaDBManager
from utils.config import OPENAI_API_KEY, GOOGLE_API_KEY
from typing import List # Required for type hinting

# Configuration (can be moved to utils.config.py later)
SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. This sentence is about a speedy fox and a relaxed dog. "
    "Chonkie is a library for text chunking, which can break down large texts into smaller pieces. "
    "Embeddings are numerical representations of text, often used in semantic search. "
    "ChromaDB is a vector database that stores these embeddings for efficient retrieval. "
    "The goal of a RAG system is to retrieve relevant information to augment LLM responses."
)
CHONKIE_CONFIG = {"chunk_size": 800, "chunk_overlap": 100} # Example, adjust as per Chonkie's DefaultChunker actual params
EMBEDDER_SERVICE = "sentence-transformers" # "openai" or "sentence-transformers"
EMBEDDER_MODEL_ST = "all-MiniLM-L6-v2" # For sentence-transformers
EMBEDDER_MODEL_OPENAI = "text-embedding-ada-002" # For OpenAI
CHUNKING_METHOD = "chapter-aware"  # Supported methods: chapter-aware, chonkie, llm

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

    # 1. Initialize Text Chunker
    print("\n--- 1. Text Chunking ---")
    try:
        # 读取三国演义文本
        text_file_path = "./三国演义.txt"
        if not os.path.exists(text_file_path):
            print(f"Error: 找不到文件 {text_file_path}")
            print("使用示例文本进行演示...")
            text_to_process = SAMPLE_TEXT
        else:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_to_process = f.read()
            print(f"成功加载三国演义文本，长度: {len(text_to_process)} 字符")
        
        if CHUNKING_METHOD == "chapter-aware":
            print("使用章节感知分块器")
            chapter_chunker = ChapterAwareChunker(
                chunk_size=CHONKIE_CONFIG["chunk_size"], 
                chunk_overlap=CHONKIE_CONFIG["chunk_overlap"],
                respect_chapter_boundaries=True,
                respect_paragraph_boundaries=True
            )
            
            # 使用章节感知分块
            chapter_chunks = chapter_chunker.chunk_text(text_to_process, metadata={"source": "三国演义"})
            
            # 3. 元数据提取
            print("\n=== 元数据提取 ===")
            metadata_extractor = MetadataExtractor(
                enable_topic_modeling=True,
                enable_ner=True,
                enable_keyword_extraction=True,
                cache_enabled=True,
                topic_model_type="lda"  # 使用LDA替代BERTopic
            )
            
            # 为章节分块添加丰富的元数据
            print("正在提取元数据...")
            enriched_chunks = metadata_extractor.enrich_chapter_metadata(chapter_chunks)
            
            # 显示元数据提取结果
            print("\n元数据提取完成！")
            for i, chunk in enumerate(enriched_chunks[:2]):  # 显示前2个章节的元数据
                print(f"\n=== 章节 {chunk.chapter_number} 元数据 ===")
                print(f"主题关键词: {chunk.metadata.get('topics', [])}")
                print(f"关键词: {chunk.metadata.get('keywords', [])[:5]}")
                print(f"人物: {chunk.metadata.get('entities', {}).get('PERSON', [])[:5]}")
                print(f"地点: {chunk.metadata.get('entities', {}).get('LOC', [])[:5]}")
                print(f"事件: {chunk.metadata.get('entities', {}).get('EVENT', [])[:3]}")
                print(f"包含对话: {chunk.metadata.get('has_dialogue', False)}")
                print(f"包含战斗: {chunk.metadata.get('has_battle', False)}")
            
            text_chunks_docs = convert_to_chonkie_documents(enriched_chunks)
            
            # 显示章节摘要
            summary = chapter_chunker.get_chapter_summary(chapter_chunks)
            print(f"章节感知分块结果:")
            print(f"  检测到章节数: {summary['total_chapters']}")
            print(f"  总分块数: {summary['total_chunks']}")
            print(f"  平均分块大小: {summary['average_chunk_size']:.0f} 字符")
            
            # 显示前几个分块的详细信息
            print("\n前3个分块信息:")
            for i, (chunk_doc, chapter_chunk) in enumerate(zip(text_chunks_docs[:3], enriched_chunks[:3])):
                print(f"\n分块 {i+1}:")
                if chapter_chunk.chapter_number:
                    print(f"  章节: 第{chapter_chunk.chapter_number}回 - {chapter_chunk.chapter_title}")
                else:
                    print(f"  章节: 序言部分")
                print(f"  类型: {chapter_chunk.chunk_type}")
                # 兼容不同的属性名：chonkie的Chunk使用text，我们的回退实现使用content
                content = getattr(chunk_doc, 'text', None) or getattr(chunk_doc, 'content', '')
                print(f"  长度: {len(content)} 字符")
                print(f"  内容预览: {content[:100].replace(chr(10), ' ')}...")
                print(f"  元数据: {chunk_doc.metadata}")
        elif CHUNKING_METHOD == "chonkie":
            print("使用传统Chonkie分块器")
            chunker = ChonkieTextChunker(chunker_name="DefaultChunker", chunker_config=CHONKIE_CONFIG)
            text_chunks_docs: List[ChonkieDocument | Chunk] = chunker.chunk_text(text_to_process, metadata={"source": "三国演义"})
            
            # 3. 元数据提取
            print("\n=== 元数据提取 ===")
            metadata_extractor = MetadataExtractor(
                enable_topic_modeling=True,
                enable_ner=True,
                enable_keyword_extraction=True,
                cache_enabled=True,
                topic_model_type="lda"  # 使用LDA替代BERTopic
            )
            
            # 为传统分块添加元数据
            print("正在提取元数据...")
            for i, chunk_doc in enumerate(text_chunks_docs[:5]):  # 只处理前5个分块作为示例
                # 兼容不同的属性名：chonkie的Chunk使用text，我们的回退实现使用content
                content = getattr(chunk_doc, 'text', None) or getattr(chunk_doc, 'content', '')
                enriched_metadata = metadata_extractor.extract_metadata(content)
                # 将ExtractedMetadata转换为字典格式
                metadata_dict = {
                    "topics": enriched_metadata.topics,
                    "entities": enriched_metadata.entities,
                    "keywords": enriched_metadata.keywords,
                    "entity_count": {k: len(v) for k, v in enriched_metadata.entities.items()},
                    "keyword_count": len(enriched_metadata.keywords)
                }
                chunk_doc.metadata.update(metadata_dict)
            
            print(f"传统分块结果: {len(text_chunks_docs)} 个分块")
            for i, chunk_doc in enumerate(text_chunks_docs[:3]):
                print(f"\n分块 {i+1}:")
                # 兼容不同的属性名：chonkie的Chunk使用text，我们的回退实现使用content
                content = getattr(chunk_doc, 'text', None) or getattr(chunk_doc, 'content', '')
                print(f"  长度: {len(content)} 字符")
                print(f"  内容预览: {content[:100]}...")
                print(f"  元数据: {chunk_doc.metadata}")
        elif CHUNKING_METHOD == "llm":
            gemini_client = GeminiClient(GOOGLE_API_KEY)
            llm_chunker = LLMChunker(gemini_client)
            text_chunks_docs = llm_chunker.chunk(text_to_process)
        else:
            raise NotImplementedError("Not supported Chunking Method.")
        
        if not text_chunks_docs:
            print("No chunks produced. Exiting.")
            return

        text_chunks_content: List[str] = [getattr(doc, 'text', None) or getattr(doc, 'content', '') for doc in text_chunks_docs]
        print(f"\n成功分块，共 {len(text_chunks_content)} 个分块")
        
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
            
            # Convert list values to strings for ChromaDB compatibility
            for key, value in list(meta.items()):
                if value is None:
                    meta[key] = ""
                elif isinstance(value, list):
                    meta[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    # Convert dict to string representation
                    meta[key] = str(value)
                elif not isinstance(value, (str, int, float, bool)):
                    # Convert any other type to string
                    meta[key] = str(value)
            
            # meta["original_text_preview"] = chunk_doc.content[:100] # Already added by ChromaDBManager
            doc_metadatas.append(meta)

        # Generate unique IDs for each chunk
        doc_ids = [f"chunk_{i}_{abs(hash((getattr(chunk_doc, 'text', None) or getattr(chunk_doc, 'content', ''))[:50]))}" for i, chunk_doc in enumerate(text_chunks_docs)]

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
