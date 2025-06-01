#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强RAG系统测试脚本

测试章节感知分块器和元数据提取功能
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_system'))

from chunking.chapter_aware_chunker import ChapterAwareChunker, convert_to_chonkie_documents
from chunking.metadata_extractor import MetadataExtractor

def test_chapter_aware_chunking():
    """测试章节感知分块功能"""
    print("=== 测试章节感知分块功能 ===")
    
    # 准备测试文本（模拟三国演义章节）
    test_text = """
第一回　宴桃园豪杰三结义　斩黄巾英雄首立功

话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚、汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。推其致乱之由，殆始于桓、灵二帝。桓帝禁锢善类，崇信宦官。及桓帝崩，灵帝即位，大将军窦武、太傅陈蕃共相辅佐。时有宦官曹节等弄权，窦武、陈蕃谋诛之，机事不密，反为所害，中涓自此愈横。

建宁二年四月望日，帝御温德殿。方升座，殿角狂风骤起，只见一条大青蛇，从梁上飞将下来，蟠于椅上。帝惊倒，左右急救入宫，百官俱奔避。须臾，蛇不见了。忽然大雷大雨，加以冰雹，落到半夜方止，坏却房屋无数。建宁四年二月，洛阳地震；又海水泛溢，沿海居民，尽被大浪卷入海中。光和元年，雌鸡化雄。六月朔，黑气十余丈，飞入温德殿中。秋七月，有虹现于玉堂；五原山岸，尽皆崩裂。种种不祥，非止一端。

第二回　张翼德怒鞭督邮　何国舅谋诛宦官

且说董卓字仲颖，陇西临洮人也，官拜河东太守，自来骄傲。当日怠慢了玄德，张飞性发，便欲杀之。玄德与关公急止之曰："他是朝廷命官，杀之不可。"飞曰："若不杀这厮，反要在他部下听令，其实不甘！"玄德曰："且再商议。"次日，接得朝廷诏书，言黄巾贼张角等造反，着各处备御。玄德看毕，叹息不已。

随后人报："门外有一将军，姓朱，名儁，字公伟，现为中郎将，奉诏讨贼，特来相请。"玄德急出迎接。朱儁见玄德器宇轩昂，心中甚喜，便请玄德同往讨贼。玄德应允。于是收拾军器，点齐人马，与朱儁同往颍川进发。
"""
    
    # 创建章节感知分块器
    chunker = ChapterAwareChunker(
        chunk_size=800,
        chunk_overlap=50,
        respect_chapter_boundaries=True,
        respect_paragraph_boundaries=True
    )
    
    # 执行分块
    print("正在执行章节感知分块...")
    chapter_chunks = chunker.chunk_text(test_text)
    
    # 显示分块结果
    print(f"\n检测到 {len(chapter_chunks)} 个章节分块")
    
    for i, chunk in enumerate(chapter_chunks):
        print(f"\n=== 分块 {i+1} ===")
        print(f"章节编号: {chunk.chapter_number}")
        print(f"章节标题: {chunk.chapter_title}")
        print(f"分块类型: {chunk.chunk_type}")
        print(f"内容长度: {len(chunk.content)} 字符")
        print(f"内容预览: {chunk.content[:100]}...")
        print(f"元数据: {chunk.metadata}")
    
    # 获取章节摘要
    summary = chunker.get_chapter_summary(chapter_chunks)
    print(f"\n=== 章节摘要 ===")
    print(f"总章节数: {summary['total_chapters']}")
    print(f"总分块数: {summary['total_chunks']}")
    print(f"平均分块大小: {summary['average_chunk_size']:.0f} 字符")
    print(f"总文本长度: {summary['total_text_length']} 字符")
    
    return chapter_chunks

def test_metadata_extraction(chapter_chunks):
    """测试元数据提取功能"""
    print("\n\n=== 测试元数据提取功能 ===")
    
    # 创建元数据提取器
    metadata_extractor = MetadataExtractor(
        enable_topic_modeling=True,
        enable_ner=True,
        enable_keyword_extraction=True,
        cache_enabled=True
    )
    
    # 为章节分块添加元数据
    print("正在提取元数据...")
    enriched_chunks = metadata_extractor.enrich_chapter_metadata(chapter_chunks)
    
    # 显示元数据提取结果
    print("\n=== 元数据提取结果 ===")
    for i, chunk in enumerate(enriched_chunks):
        print(f"\n=== 章节 {chunk.chapter_number or i+1} 元数据 ===")
        print(f"主题关键词: {chunk.metadata.get('topics', [])}")
        print(f"关键词: {chunk.metadata.get('keywords', [])[:8]}")
        
        entities = chunk.metadata.get('entities', {})
        print(f"人物: {entities.get('PERSON', [])[:5]}")
        print(f"地点: {entities.get('LOC', [])[:5]}")
        print(f"组织: {entities.get('ORG', [])[:3]}")
        print(f"事件: {entities.get('EVENT', [])[:3]}")
        
        print(f"实体统计: {chunk.metadata.get('entity_count', {})}")
        print(f"关键词数量: {chunk.metadata.get('keyword_count', 0)}")
        print(f"内容长度: {chunk.metadata.get('content_length', 0)}")
        print(f"包含对话: {chunk.metadata.get('has_dialogue', False)}")
        print(f"包含战斗: {chunk.metadata.get('has_battle', False)}")
    
    # 显示缓存统计
    cache_stats = metadata_extractor.get_cache_stats()
    print(f"\n=== 缓存统计 ===")
    print(f"缓存大小: {cache_stats['cache_size']}")
    print(f"缓存启用: {cache_stats['cache_enabled']}")
    
    return enriched_chunks

def test_chonkie_conversion(enriched_chunks):
    """测试转换为ChonkieDocument格式"""
    print("\n\n=== 测试ChonkieDocument转换 ===")
    
    # 转换为ChonkieDocument格式
    chonkie_docs = convert_to_chonkie_documents(enriched_chunks)
    
    print(f"转换完成，共 {len(chonkie_docs)} 个ChonkieDocument")
    
    # 显示转换结果
    for i, doc in enumerate(chonkie_docs[:2]):  # 只显示前2个
        print(f"\n=== ChonkieDocument {i+1} ===")
        # 兼容不同的属性名：chonkie的Chunk使用text，我们的回退实现使用content
        content = getattr(doc, 'text', None) or getattr(doc, 'content', '')
        print(f"内容长度: {len(content)} 字符")
        print(f"内容预览: {content[:100]}...")
        print(f"元数据键: {list(doc.metadata.keys())}")
        print(f"章节信息: 第{doc.metadata.get('chapter_number', 'N/A')}回")
        print(f"主题: {doc.metadata.get('topics', [])[:3]}")
        print(f"人物: {doc.metadata.get('entities', {}).get('PERSON', [])[:3]}")
    
    return chonkie_docs

def main():
    """主测试函数"""
    print("开始测试增强RAG系统...\n")
    
    try:
        # 测试章节感知分块
        chapter_chunks = test_chapter_aware_chunking()
        
        # 测试元数据提取
        enriched_chunks = test_metadata_extraction(chapter_chunks)
        
        # 测试格式转换
        chonkie_docs = test_chonkie_conversion(enriched_chunks)
        
        print("\n\n=== 测试完成 ===")
        print("✅ 章节感知分块功能正常")
        print("✅ 元数据提取功能正常")
        print("✅ ChonkieDocument转换功能正常")
        print("\n增强RAG系统测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)