"""章节感知的文本分块器，专门用于处理《三国演义》等章回体小说。

该模块提供ChapterAwareChunker类，能够：
1. 识别章节边界（如"第X回"）
2. 按章节进行智能分块
3. 提取丰富的元数据信息
4. 与chonkie库的RecursiveChunker集成
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re

from chunking.chonkie_chunker import ChonkieDocument, CHONKIE_AVAILABLE

try:
    from chonkie import RecursiveChunker
except ImportError:
    pass


@dataclass
class ChapterChunk:
    """表示一个章节分块的数据结构"""
    content: str
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    chunk_type: str = "chapter"  # "prologue", "chapter", "epilogue"
    start_position: int = 0
    end_position: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ChapterAwareChunker:
    """章节感知的文本分块器"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 respect_chapter_boundaries: bool = True,
                 respect_paragraph_boundaries: bool = True,
                 chapter_pattern: str = r"第[一二三四五六七八九十百千万\d]+回"):
        """
        初始化章节感知分块器
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            respect_chapter_boundaries: 是否尊重章节边界
            respect_paragraph_boundaries: 是否尊重段落边界
            chapter_pattern: 章节标题的正则表达式模式
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_chapter_boundaries = respect_chapter_boundaries
        self.respect_paragraph_boundaries = respect_paragraph_boundaries
        self.chapter_pattern = chapter_pattern
        
        # 初始化RecursiveChunker（如果可用）
        if CHONKIE_AVAILABLE:
            self.recursive_chunker = RecursiveChunker(
                chunk_size=chunk_size,
                # 注意：RecursiveChunker可能不支持chunk_overlap参数
            )
        else:
            self.recursive_chunker = None
    
    def detect_chapters(self, text: str) -> List[Tuple[int, int, str, Optional[int]]]:
        """
        检测文本中的章节边界
        
        Args:
            text: 输入文本
            
        Returns:
            List of (start_pos, end_pos, title, chapter_num) tuples
        """
        chapters = []
        
        # 查找所有章节标题
        pattern = re.compile(self.chapter_pattern + r"[^\n]*")
        matches = list(pattern.finditer(text))
        
        if not matches:
            # 如果没有找到章节，将整个文本作为一个块
            return [(0, len(text), "全文", None)]
        
        # 处理序言（第一章之前的内容）
        first_chapter_start = matches[0].start()
        if first_chapter_start > 0:
            prologue_content = text[:first_chapter_start].strip()
            if prologue_content:
                chapters.append((0, first_chapter_start, "序言", None))
        
        # 处理各章节
        for i, match in enumerate(matches):
            title = match.group().strip()
            start_pos = match.start()
            
            # 提取章节号
            chapter_num = self._extract_chapter_number(title)
            
            # 确定章节结束位置
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            chapters.append((start_pos, end_pos, title, chapter_num))
        
        return chapters
    
    def _extract_chapter_number(self, title: str) -> Optional[int]:
        """从章节标题中提取章节号"""
        # 匹配数字
        num_match = re.search(r'\d+', title)
        if num_match:
            return int(num_match.group())
        
        # 匹配中文数字
        chinese_nums = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
            '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
            # 可以继续扩展...
        }
        
        for chinese, num in chinese_nums.items():
            if chinese in title:
                return num
        
        return None
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChapterChunk]:
        """
        对文本进行章节感知分块
        
        Args:
            text: 输入文本
            metadata: 额外的元数据
            
        Returns:
            章节分块列表
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        chapters = self.detect_chapters(text)
        
        for start_pos, end_pos, title, chapter_num in chapters:
            chapter_content = text[start_pos:end_pos].strip()
            
            if not chapter_content:
                continue
            
            # 确定分块类型
            if chapter_num is None:
                chunk_type = "prologue" if "序言" in title else "other"
            else:
                chunk_type = "chapter"
            
            # 如果章节内容太长，需要进一步分块
            if len(chapter_content) > self.chunk_size and self.recursive_chunker:
                sub_chunks = self._split_long_chapter(chapter_content, title, chapter_num, start_pos, metadata)
                chunks.extend(sub_chunks)
            else:
                # 创建单个章节分块
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chapter_number": chapter_num,
                    "chapter_title": title,
                    "chunk_type": chunk_type,
                    "start_position": start_pos,
                    "end_position": end_pos,
                    "original_text": chapter_content
                })
                
                chunk = ChapterChunk(
                    content=chapter_content,
                    chapter_number=chapter_num,
                    chapter_title=title,
                    chunk_type=chunk_type,
                    start_position=start_pos,
                    end_position=end_pos,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_long_chapter(self, content: str, title: str, chapter_num: Optional[int], 
                           base_start_pos: int, base_metadata: Dict[str, Any]) -> List[ChapterChunk]:
        """将过长的章节进一步分块"""
        chunks = []
        
        if self.recursive_chunker and CHONKIE_AVAILABLE:
            # 使用RecursiveChunker进行子分块
            try:
                sub_chunks = self.recursive_chunker.chunk(content)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chapter_number": chapter_num,
                        "chapter_title": title,
                        "chunk_type": "chapter_part",
                        "sub_chunk_index": i,
                        "start_position": base_start_pos,
                        "original_text": sub_chunk.text if hasattr(sub_chunk, 'text') else str(sub_chunk)
                    })
                    
                    chunk = ChapterChunk(
                        content=sub_chunk.text if hasattr(sub_chunk, 'text') else str(sub_chunk),
                        chapter_number=chapter_num,
                        chapter_title=title,
                        chunk_type="chapter_part",
                        start_position=base_start_pos,
                        end_position=base_start_pos + len(content),
                        metadata=chunk_metadata
                    )
                    chunks.append(chunk)
            except Exception as e:
                print(f"RecursiveChunker failed, using fallback: {e}")
                # 回退到简单分块
                chunks = self._simple_split_chapter(content, title, chapter_num, base_start_pos, base_metadata)
        else:
            # 使用简单的分块策略
            chunks = self._simple_split_chapter(content, title, chapter_num, base_start_pos, base_metadata)
        
        return chunks
    
    def _simple_split_chapter(self, content: str, title: str, chapter_num: Optional[int],
                             base_start_pos: int, base_metadata: Dict[str, Any]) -> List[ChapterChunk]:
        """简单的章节分块策略（回退方案）"""
        chunks = []
        
        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # 创建当前分块
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chapter_number": chapter_num,
                    "chapter_title": title,
                    "chunk_type": "chapter_part",
                    "sub_chunk_index": chunk_index,
                    "start_position": base_start_pos,
                    "original_text": current_chunk
                })
                
                chunk = ChapterChunk(
                    content=current_chunk,
                    chapter_number=chapter_num,
                    chapter_title=title,
                    chunk_type="chapter_part",
                    start_position=base_start_pos,
                    end_position=base_start_pos + len(content),
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
                
                # 重置当前分块
                current_chunk = para
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # 处理最后一个分块
        if current_chunk:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chapter_number": chapter_num,
                "chapter_title": title,
                "chunk_type": "chapter_part",
                "sub_chunk_index": chunk_index,
                "start_position": base_start_pos,
                "original_text": current_chunk
            })
            
            chunk = ChapterChunk(
                content=current_chunk,
                chapter_number=chapter_num,
                chapter_title=title,
                chunk_type="chapter_part",
                start_position=base_start_pos,
                end_position=base_start_pos + len(content),
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chapter_summary(self, chunks: List[ChapterChunk]) -> Dict[str, Any]:
        """获取章节分块的统计摘要"""
        total_chunks = len(chunks)
        chapters = set()
        total_length = 0
        
        for chunk in chunks:
            if chunk.chapter_number:
                chapters.add(chunk.chapter_number)
            total_length += len(chunk.content)
        
        return {
            "total_chapters": len(chapters),
            "total_chunks": total_chunks,
            "average_chunk_size": total_length / total_chunks if total_chunks > 0 else 0,
            "total_text_length": total_length
        }


def convert_to_chonkie_documents(chapter_chunks: List[ChapterChunk]) -> List[ChonkieDocument]:
    """将ChapterChunk转换为ChonkieDocument格式"""
    documents = []
    
    for chunk in chapter_chunks:
        # 根据chonkie库是否可用，使用不同的构造方式
        if CHONKIE_AVAILABLE:
            # 如果chonkie库可用，使用其Chunk类的构造方式
            from chonkie.types import Chunk
            # 从metadata中获取位置信息，如果没有则使用默认值
            start_index = chunk.metadata.get('start_offset', 0)
            end_index = chunk.metadata.get('end_offset', len(chunk.content))
            token_count = len(chunk.content.split())  # 简单的token计数
            
            doc = Chunk(
                text=chunk.content,
                start_index=start_index,
                end_index=end_index,
                token_count=token_count
            )
            # 单独设置metadata属性
            doc.metadata = chunk.metadata
        else:
            # 如果使用我们的回退实现，使用content参数
            doc = ChonkieDocument(content=chunk.content, metadata=chunk.metadata)
        documents.append(doc)
    
    return documents