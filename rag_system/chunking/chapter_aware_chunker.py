#!/usr/bin/env python3
"""
章节感知的智能分块器

专门针对古典小说（如三国演义）的章节结构进行智能分块，
能够准确识别章节边界，保持故事情节的完整性。
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class ChapterChunk:
    """章节分块数据结构"""
    content: str
    metadata: Dict[str, Any]
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    chunk_type: str = "content"  # "title", "content", "mixed"
    start_position: int = 0
    end_position: int = 0


class ChapterAwareChunker:
    """章节感知的智能分块器
    
    特点：
    1. 自动识别章节标题格式
    2. 按章节边界进行分块
    3. 保持故事情节完整性
    4. 支持灵活的分块策略
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 respect_chapter_boundaries: bool = True,
                 respect_paragraph_boundaries: bool = True,
                 chapter_pattern: Optional[str] = None):
        """
        初始化章节感知分块器
        
        Args:
            chunk_size: 目标分块大小（字符数）
            chunk_overlap: 分块重叠大小
            respect_chapter_boundaries: 是否尊重章节边界
            respect_paragraph_boundaries: 是否尊重段落边界
            chapter_pattern: 自定义章节标题正则表达式
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_chapter_boundaries = respect_chapter_boundaries
        self.respect_paragraph_boundaries = respect_paragraph_boundaries
        
        # 默认章节标题模式（适用于三国演义等古典小说）
        self.chapter_pattern = chapter_pattern or r'第[一二三四五六七八九十百零]+回\s+.+'
        self.chapter_regex = re.compile(self.chapter_pattern)
        
        # 数字转换映射（中文数字到阿拉伯数字）
        self.chinese_numbers = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '零': 0, '百': 100
        }
    
    def _convert_chinese_number(self, chinese_num: str) -> int:
        """将中文数字转换为阿拉伯数字"""
        try:
            if '十' in chinese_num:
                if chinese_num == '十':
                    return 10
                elif chinese_num.startswith('十'):
                    return 10 + self.chinese_numbers.get(chinese_num[1], 0)
                elif chinese_num.endswith('十'):
                    return self.chinese_numbers.get(chinese_num[0], 0) * 10
                else:
                    parts = chinese_num.split('十')
                    return self.chinese_numbers.get(parts[0], 0) * 10 + self.chinese_numbers.get(parts[1], 0)
            elif '百' in chinese_num:
                if len(chinese_num) == 1:
                    return 100
                else:
                    # 处理如"一百二十三"的情况
                    parts = chinese_num.split('百')
                    hundred = self.chinese_numbers.get(parts[0], 1) * 100
                    if len(parts) > 1 and parts[1]:
                        hundred += self._convert_chinese_number(parts[1])
                    return hundred
            else:
                return self.chinese_numbers.get(chinese_num, 0)
        except:
            return 0
    
    def _extract_chapter_info(self, title: str) -> Tuple[int, str]:
        """从章节标题中提取章节号和标题"""
        match = re.match(r'第([一二三四五六七八九十百零]+)回\s+(.+)', title.strip())
        if match:
            chinese_num = match.group(1)
            chapter_title = match.group(2)
            chapter_number = self._convert_chinese_number(chinese_num)
            return chapter_number, chapter_title
        return 0, title.strip()
    
    def _find_chapter_boundaries(self, text: str) -> List[Dict[str, Any]]:
        """找到所有章节边界"""
        chapters = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if self.chapter_regex.match(line):
                chapter_number, chapter_title = self._extract_chapter_info(line)
                chapters.append({
                    'line_number': i,
                    'title': line,
                    'chapter_number': chapter_number,
                    'chapter_title': chapter_title,
                    'start_pos': sum(len(lines[j]) + 1 for j in range(i))  # +1 for newline
                })
        
        return chapters
    
    def _split_by_chapters(self, text: str) -> List[ChapterChunk]:
        """按章节分割文本"""
        chapters = self._find_chapter_boundaries(text)
        chunks = []
        
        if not chapters:
            # 如果没有找到章节，按普通方式分块
            return self._split_by_size(text)
        
        for i, chapter in enumerate(chapters):
            # 确定章节内容的起始和结束位置
            start_pos = chapter['start_pos']
            if i + 1 < len(chapters):
                end_pos = chapters[i + 1]['start_pos']
            else:
                end_pos = len(text)
            
            chapter_content = text[start_pos:end_pos].strip()
            
            # 如果章节内容太长，需要进一步分块
            if len(chapter_content) <= self.chunk_size:
                chunks.append(ChapterChunk(
                    content=chapter_content,
                    metadata={
                        'chapter_number': chapter['chapter_number'],
                        'chapter_title': chapter['chapter_title'],
                        'full_title': chapter['title'],
                        'chunk_index': 0,
                        'total_chunks_in_chapter': 1
                    },
                    chapter_number=chapter['chapter_number'],
                    chapter_title=chapter['chapter_title'],
                    chunk_type='mixed',
                    start_position=start_pos,
                    end_position=end_pos
                ))
            else:
                # 章节内容太长，需要分割
                chapter_chunks = self._split_chapter_content(
                    chapter_content, 
                    chapter['chapter_number'], 
                    chapter['chapter_title'],
                    chapter['title']
                )
                chunks.extend(chapter_chunks)
        
        return chunks
    
    def _split_chapter_content(self, content: str, chapter_number: int, 
                              chapter_title: str, full_title: str) -> List[ChapterChunk]:
        """分割单个章节的内容"""
        chunks = []
        
        if self.respect_paragraph_boundaries:
            # 按段落边界分割
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                # 检查添加这个段落是否会超过大小限制
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:  # +2 for \n\n
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # 保存当前块
                    if current_chunk:
                        chunks.append(ChapterChunk(
                            content=current_chunk,
                            metadata={
                                'chapter_number': chapter_number,
                                'chapter_title': chapter_title,
                                'full_title': full_title,
                                'chunk_index': chunk_index,
                                'total_chunks_in_chapter': -1  # 稍后更新
                            },
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            chunk_type='content'
                        ))
                        chunk_index += 1
                    
                    # 开始新块
                    current_chunk = para
            
            # 添加最后一个块
            if current_chunk:
                chunks.append(ChapterChunk(
                    content=current_chunk,
                    metadata={
                        'chapter_number': chapter_number,
                        'chapter_title': chapter_title,
                        'full_title': full_title,
                        'chunk_index': chunk_index,
                        'total_chunks_in_chapter': -1
                    },
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    chunk_type='content'
                ))
        else:
            # 按固定大小分割
            chunks = self._split_by_size_with_metadata(content, chapter_number, chapter_title, full_title)
        
        # 更新总块数
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata['total_chunks_in_chapter'] = total_chunks
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[ChapterChunk]:
        """按固定大小分割文本（回退方法）"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]
            
            chunks.append(ChapterChunk(
                content=chunk_content,
                metadata={
                    'chunk_index': chunk_index,
                    'start_position': start,
                    'end_position': end,
                    'chunk_type': 'size_based'
                },
                chunk_type='content',
                start_position=start,
                end_position=end
            ))
            
            # 计算下一个块的起始位置（考虑重叠）
            if self.chunk_overlap > 0 and end < len(text):
                start = end - self.chunk_overlap
            else:
                start = end
            
            chunk_index += 1
        
        return chunks
    
    def _split_by_size_with_metadata(self, content: str, chapter_number: int, 
                                   chapter_title: str, full_title: str) -> List[ChapterChunk]:
        """按固定大小分割，但保留章节元数据"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            chunks.append(ChapterChunk(
                content=chunk_content,
                metadata={
                    'chapter_number': chapter_number,
                    'chapter_title': chapter_title,
                    'full_title': full_title,
                    'chunk_index': chunk_index,
                    'start_position': start,
                    'end_position': end
                },
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                chunk_type='content',
                start_position=start,
                end_position=end
            ))
            
            if self.chunk_overlap > 0 and end < len(content):
                start = end - self.chunk_overlap
            else:
                start = end
            
            chunk_index += 1
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[ChapterChunk]:
        """主要的分块方法"""
        if not text.strip():
            return []
        
        base_metadata = metadata or {}
        
        if self.respect_chapter_boundaries:
            chunks = self._split_by_chapters(text)
        else:
            chunks = self._split_by_size(text)
        
        # 添加基础元数据
        for chunk in chunks:
            chunk.metadata.update(base_metadata)
            chunk.metadata['chunker_type'] = 'chapter_aware'
            chunk.metadata['chunk_size_config'] = self.chunk_size
            chunk.metadata['chunk_overlap_config'] = self.chunk_overlap
        
        return chunks
    
    def get_chapter_summary(self, chunks: List[ChapterChunk]) -> Dict[str, Any]:
        """获取分块结果的章节摘要"""
        chapters = {}
        total_chunks = len(chunks)
        
        for chunk in chunks:
            chapter_num = chunk.chapter_number or 0
            if chapter_num not in chapters:
                chapters[chapter_num] = {
                    'chapter_number': chapter_num,
                    'chapter_title': chunk.chapter_title or 'Unknown',
                    'chunk_count': 0,
                    'total_length': 0
                }
            
            chapters[chapter_num]['chunk_count'] += 1
            chapters[chapter_num]['total_length'] += len(chunk.content)
        
        return {
            'total_chunks': total_chunks,
            'total_chapters': len(chapters),
            'chapters': list(chapters.values()),
            'average_chunk_size': sum(len(chunk.content) for chunk in chunks) / total_chunks if total_chunks > 0 else 0
        }


# 兼容性适配器，使其与现有的ChonkieDocument接口兼容
class ChonkieDocument:
    """兼容性文档类"""
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}


def convert_to_chonkie_documents(chunks: List[ChapterChunk]) -> List[ChonkieDocument]:
    """将ChapterChunk转换为ChonkieDocument格式"""
    return [
        ChonkieDocument(content=chunk.content, metadata=chunk.metadata)
        for chunk in chunks
    ]


# 示例使用
if __name__ == "__main__":
    # 测试代码
    sample_text = """
第一回 宴桃园豪杰三结义 斩黄巾英雄首立功

滚滚长江东逝水，浪花淘尽英雄。是非成败转头空。
青山依旧在，几度夕阳红。

话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。

第二回 张翼德怒鞭督邮 何国舅谋诛宦竖

却说董卓字仲颖，陇西临洮人也，官拜河东太守。
"""
    
    chunker = ChapterAwareChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text)
    
    print(f"分块结果：共 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1}:")
        print(f"章节: {chunk.chapter_number} - {chunk.chapter_title}")
        print(f"类型: {chunk.chunk_type}")
        print(f"内容: {chunk.content[:100]}...")
        print(f"元数据: {chunk.metadata}")
    
    summary = chunker.get_chapter_summary(chunks)
    print(f"\n摘要: {summary}")