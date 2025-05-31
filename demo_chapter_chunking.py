#!/usr/bin/env python3
"""
章节感知分块演示脚本

演示如何使用章节感知分块器来准确分解三国演义的章节结构
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from rag_system.chunking.chapter_aware_chunker import ChapterAwareChunker, convert_to_chonkie_documents


def load_text_file(file_path: str) -> str:
    """加载文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"加载文件失败: {e}")
        return ""


def analyze_chapter_structure(text: str, chunker: ChapterAwareChunker):
    """分析章节结构"""
    print("=== 章节结构分析 ===")
    
    # 找到所有章节
    chapters = chunker._find_chapter_boundaries(text)
    print(f"检测到 {len(chapters)} 个章节:")
    
    for i, chapter in enumerate(chapters[:10]):  # 只显示前10个章节
        print(f"  {i+1:2d}. 第{chapter['chapter_number']:3d}回 {chapter['chapter_title']}")
    
    if len(chapters) > 10:
        print(f"  ... 还有 {len(chapters) - 10} 个章节")
    
    return chapters


def demonstrate_chunking_strategies():
    """演示不同的分块策略"""
    print("\n=== 分块策略演示 ===")
    
    # 加载三国演义文本
    text_file = "三国演义.txt"
    if not os.path.exists(text_file):
        print(f"错误：找不到文件 {text_file}")
        return
    
    text = load_text_file(text_file)
    if not text:
        print("无法加载文本内容")
        return
    
    print(f"文本总长度: {len(text):,} 字符")
    
    # 策略1：章节感知分块（推荐）
    print("\n--- 策略1：章节感知分块 ---")
    chunker1 = ChapterAwareChunker(
        chunk_size=800,
        chunk_overlap=100,
        respect_chapter_boundaries=True,
        respect_paragraph_boundaries=True
    )
    
    # 分析章节结构
    chapters = analyze_chapter_structure(text, chunker1)
    
    # 执行分块
    chunks1 = chunker1.chunk_text(text)
    summary1 = chunker1.get_chapter_summary(chunks1)
    
    print(f"\n分块结果:")
    print(f"  总块数: {summary1['total_chunks']}")
    print(f"  章节数: {summary1['total_chapters']}")
    print(f"  平均块大小: {summary1['average_chunk_size']:.0f} 字符")
    
    # 显示前几个块的详细信息
    print("\n前5个分块示例:")
    for i, chunk in enumerate(chunks1[:5]):
        print(f"\n块 {i+1}:")
        print(f"  章节: 第{chunk.chapter_number}回 - {chunk.chapter_title}")
        print(f"  类型: {chunk.chunk_type}")
        print(f"  长度: {len(chunk.content)} 字符")
        print(f"  内容预览: {chunk.content[:100].replace(chr(10), ' ')}...")
        if 'chunk_index' in chunk.metadata:
            print(f"  章节内块序号: {chunk.metadata['chunk_index'] + 1}/{chunk.metadata['total_chunks_in_chapter']}")
    
    # 策略2：传统固定大小分块（对比）
    print("\n--- 策略2：传统固定大小分块（对比） ---")
    chunker2 = ChapterAwareChunker(
        chunk_size=800,
        chunk_overlap=100,
        respect_chapter_boundaries=False,
        respect_paragraph_boundaries=False
    )
    
    chunks2 = chunker2.chunk_text(text)
    summary2 = chunker2.get_chapter_summary(chunks2)
    
    print(f"分块结果:")
    print(f"  总块数: {summary2['total_chunks']}")
    print(f"  平均块大小: {summary2['average_chunk_size']:.0f} 字符")
    
    # 策略3：仅尊重段落边界
    print("\n--- 策略3：仅尊重段落边界 ---")
    chunker3 = ChapterAwareChunker(
        chunk_size=800,
        chunk_overlap=100,
        respect_chapter_boundaries=False,
        respect_paragraph_boundaries=True
    )
    
    chunks3 = chunker3.chunk_text(text)
    summary3 = chunker3.get_chapter_summary(chunks3)
    
    print(f"分块结果:")
    print(f"  总块数: {summary3['total_chunks']}")
    print(f"  平均块大小: {summary3['average_chunk_size']:.0f} 字符")
    
    # 比较分析
    print("\n=== 策略比较分析 ===")
    print(f"{'策略':<20} {'总块数':<10} {'平均大小':<10} {'章节完整性':<12}")
    print("-" * 60)
    print(f"{'章节感知分块':<20} {summary1['total_chunks']:<10} {summary1['average_chunk_size']:<10.0f} {'优秀':<12}")
    print(f"{'固定大小分块':<20} {summary2['total_chunks']:<10} {summary2['average_chunk_size']:<10.0f} {'较差':<12}")
    print(f"{'段落边界分块':<20} {summary3['total_chunks']:<10} {summary3['average_chunk_size']:<10.0f} {'一般':<12}")
    
    return chunks1, summary1


def demonstrate_chapter_integrity(chunks, summary):
    """演示章节完整性保护"""
    print("\n=== 章节完整性演示 ===")
    
    # 按章节分组显示
    chapters_dict = {}
    for chunk in chunks:
        chapter_num = chunk.chapter_number or 0
        if chapter_num not in chapters_dict:
            chapters_dict[chapter_num] = []
        chapters_dict[chapter_num].append(chunk)
    
    # 显示前几个章节的分块情况
    print("前5个章节的分块情况:")
    for chapter_num in sorted(chapters_dict.keys())[:5]:
        chapter_chunks = chapters_dict[chapter_num]
        if chapter_num == 0:
            print(f"\n序言部分: {len(chapter_chunks)} 个块")
        else:
            first_chunk = chapter_chunks[0]
            print(f"\n第{chapter_num}回 {first_chunk.chapter_title}: {len(chapter_chunks)} 个块")
        
        for i, chunk in enumerate(chapter_chunks):
            content_preview = chunk.content[:50].replace('\n', ' ').strip()
            print(f"  块{i+1}: {len(chunk.content)}字符 - {content_preview}...")


def demonstrate_search_optimization(chunks):
    """演示搜索优化"""
    print("\n=== 搜索优化演示 ===")
    
    # 模拟搜索场景
    search_queries = [
        "刘备",
        "桃园结义",
        "诸葛亮",
        "赤壁之战",
        "关羽"
    ]
    
    print("模拟搜索结果（基于章节元数据）:")
    
    for query in search_queries:
        print(f"\n搜索: '{query}'")
        relevant_chunks = []
        
        for chunk in chunks:
            if query in chunk.content:
                relevant_chunks.append(chunk)
        
        if relevant_chunks:
            print(f"  找到 {len(relevant_chunks)} 个相关块:")
            for chunk in relevant_chunks[:3]:  # 只显示前3个
                if chunk.chapter_number:
                    print(f"    - 第{chunk.chapter_number}回: {chunk.chapter_title}")
                else:
                    print(f"    - 序言部分")
        else:
            print(f"  未找到相关内容")


def save_chunking_results(chunks, output_file="chunking_results.txt"):
    """保存分块结果到文件"""
    print(f"\n=== 保存分块结果到 {output_file} ===")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("三国演义章节感知分块结果\n")
            f.write("=" * 50 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"块 {i+1}\n")
                f.write("-" * 20 + "\n")
                f.write(f"章节: 第{chunk.chapter_number}回 - {chunk.chapter_title}\n")
                f.write(f"类型: {chunk.chunk_type}\n")
                f.write(f"长度: {len(chunk.content)} 字符\n")
                f.write(f"元数据: {chunk.metadata}\n")
                f.write(f"内容:\n{chunk.content}\n")
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"分块结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存失败: {e}")


def main():
    """主函数"""
    print("三国演义章节感知分块演示")
    print("=" * 50)
    
    # 演示不同分块策略
    chunks, summary = demonstrate_chunking_strategies()
    
    if chunks:
        # 演示章节完整性
        demonstrate_chapter_integrity(chunks, summary)
        
        # 演示搜索优化
        demonstrate_search_optimization(chunks)
        
        # 保存结果（可选）
        save_choice = input("\n是否保存分块结果到文件？(y/n): ").strip().lower()
        if save_choice == 'y':
            save_chunking_results(chunks)
        
        print("\n=== 总结 ===")
        print("章节感知分块器的优势:")
        print("1. 自动识别章节边界，保持故事完整性")
        print("2. 智能处理中文数字章节标题")
        print("3. 灵活的分块策略，适应不同需求")
        print("4. 丰富的元数据，支持精确检索")
        print("5. 兼容现有RAG系统架构")
        
        print("\n推荐配置:")
        print("- chunk_size: 800-1200 字符")
        print("- chunk_overlap: 100-200 字符")
        print("- respect_chapter_boundaries: True")
        print("- respect_paragraph_boundaries: True")


if __name__ == "__main__":
    main()