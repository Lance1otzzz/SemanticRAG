#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块方法对比测试脚本
对比章节感知分块器与传统分块方法的效果
"""

import os
import sys
import time
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system.chunking.chapter_aware_chunker import ChapterAwareChunker
from rag_system.chunking.chonkie_chunker import ChonkieTextChunker, ChonkieDocument

def load_text() -> str:
    """加载三国演义文本"""
    file_path = "三国演义.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return ""
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return ""

def test_chapter_aware_chunker(text: str) -> List[ChonkieDocument]:
    """测试章节感知分块器"""
    chunker = ChapterAwareChunker(
        chunk_size=800,
        chunk_overlap=100,
        respect_chapter_boundaries=True,
        respect_paragraph_boundaries=True
    )
    chapter_chunks = chunker.chunk_text(text)
    # 转换为ChonkieDocument格式
    from rag_system.chunking.chapter_aware_chunker import convert_to_chonkie_documents
    return convert_to_chonkie_documents(chapter_chunks)

def test_fixed_size_chunker(text: str) -> List[ChonkieDocument]:
    """测试固定大小分块器"""
    chunk_size = 800
    overlap = 100
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text.strip():
            chunks.append(ChonkieDocument(
                content=chunk_text,
                metadata={'chunk_index': len(chunks)}
            ))
    
    return chunks

def test_chonkie_chunker(text: str) -> List[ChonkieDocument]:
    """测试Chonkie分块器"""
    chunker = ChonkieTextChunker(
        chunker_config={
            'chunk_size': 800,
            'chunk_overlap': 100
        }
    )
    return chunker.chunk_text(text)

def analyze_chunk_quality(chunks: List[ChonkieDocument]) -> Dict[str, Any]:
    """分析分块质量"""
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'boundary_violations': 0
        }
    
    # 检查章节边界保护
    boundary_violations = 0
    for chunk in chunks:
        # 处理不同类型的chunk对象
        if hasattr(chunk, 'content'):
            content = chunk.content
        elif hasattr(chunk, 'text'):
            content = chunk.text
        else:
            content = str(chunk)
            
        # 简单检查：如果分块在句子中间结束（不以句号、问号、感叹号结尾）
        if content and not content.rstrip().endswith(('。', '？', '！', '"')):
            boundary_violations += 1
    
    def get_content(chunk):
        if hasattr(chunk, 'content'):
            return chunk.content
        elif hasattr(chunk, 'text'):
            return chunk.text
        else:
            return str(chunk)
    
    return {
        'total_chunks': len(chunks),
        'avg_chunk_size': sum(len(get_content(chunk)) for chunk in chunks) / len(chunks) if chunks else 0,
        'boundary_violations': boundary_violations
    }

def main():
    """主函数：运行分块方法对比测试"""
    print("=" * 60)
    print("分块方法对比测试")
    print("=" * 60)
    
    # 加载文本
    text = load_text()
    if not text:
        print("无法加载文本文件，测试终止")
        return
    
    print(f"文本长度: {len(text)} 字符")
    print()
    
    # 测试不同的分块方法
    methods = {
        "章节感知分块器": lambda: test_chapter_aware_chunker(text),
        "固定大小分块器": lambda: test_fixed_size_chunker(text),
        "Chonkie分块器": lambda: test_chonkie_chunker(text)
    }
    
    results = {}
    for method_name, method_func in methods.items():
        print(f"\n测试 {method_name}...")
        try:
            start_time = time.time()
            chunks = method_func()
            end_time = time.time()
            
            # 分析分块质量
            quality = analyze_chunk_quality(chunks)
            quality['processing_time'] = end_time - start_time
            
            results[method_name] = {
                'chunks': chunks,
                'quality': quality
            }
            
            print(f"  分块数量: {quality['total_chunks']}")
            print(f"  平均大小: {quality['avg_chunk_size']:.1f} 字符")
            print(f"  处理时间: {quality['processing_time']:.3f} 秒")
            print(f"  边界违规: {quality['boundary_violations']}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results[method_name] = None
    
    # 详细对比分析
    print("\n" + "=" * 60)
    print("详细对比分析")
    print("=" * 60)
    
    # 章节完整性测试
    if "章节感知分块器" in results and results["章节感知分块器"]:
        chapter_chunks = results["章节感知分块器"]['chunks']
        print("\n章节完整性测试:")
        chapter_titles = [chunk.metadata.get('chapter_title', '未知') 
                         for chunk in chapter_chunks 
                         if chunk.metadata.get('chapter_title')]
        print(f"  包含章节标题的分块: {len(set(chapter_titles))} 个不同章节")
        print(f"  示例章节: {list(set(chapter_titles))[:3]}")
    
    # 搜索效果测试
    print("\n搜索效果测试:")
    test_queries = ["赤壁之战", "关羽", "诸葛亮"]
    
    def get_chunk_content(chunk):
        """获取chunk的内容"""
        if hasattr(chunk, 'content'):
            return chunk.content
        elif hasattr(chunk, 'text'):
            return chunk.text
        else:
            return str(chunk)
    
    for query in test_queries:
        print(f"\n搜索关键词: '{query}'")
        for method_name, result in results.items():
            if result:
                chunks = result['chunks']
                relevant_chunks = [chunk for chunk in chunks 
                                 if query in get_chunk_content(chunk)]
                print(f"  {method_name}: {len(relevant_chunks)} 个相关分块")
                
                if relevant_chunks and method_name == "章节感知分块器":
                    # 显示章节信息
                    chapters = set(chunk.metadata.get('chapter_title', '未知') 
                                 for chunk in relevant_chunks)
                    print(f"    涉及章节: {list(chapters)[:2]}")
    
    # 推荐配置
    print("\n" + "=" * 60)
    print("推荐使用章节感知分块器")
    print("=" * 60)
    print("优势:")
    print("1. 自动识别章节边界，保持故事完整性")
    print("2. 支持中文数字章节标题")
    print("3. 灵活的分块策略")
    print("4. 丰富的元数据信息")
    print("5. 与现有RAG系统架构兼容")
    print("\n推荐配置:")
    print("- chunk_size: 800-1200 字符")
    print("- chunk_overlap: 100-200 字符")
    print("- respect_chapter_boundaries: True")
    print("- respect_paragraph_boundaries: True")

if __name__ == "__main__":
    main()