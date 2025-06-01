#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题建模方法对比演示
比较BERTopic和LDA在中文文本处理上的效果
"""

import os
import sys
sys.path.append('./rag_system')

from rag_system.chunking.metadata_extractor import MetadataExtractor

def test_topic_models():
    """测试不同的主题建模方法"""
    
    # 准备测试文本
    sample_texts = [
        """刘备字玄德，涿郡涿县人，汉景帝子中山靖王刘胜之后也。胜子贞，元狩六年封涿县陆城亭侯，坐酎金失侯，因家焉。先主祖雄，父弘，世仕州郡。雄举孝廉，官至东郡范令。先主少孤，与母贩履织席为业。舍东南角篱上有桑树生高五丈余，遥望见童童如小车盖，往来者皆怪此树非凡，或谓当出贵人。先主少时，与宗中诸小儿于树下戏，言：吾必当乘此羽葆盖车。叔父子敬谓曰：汝勿妄语，灭吾门也！""",
        
        """关羽字云长，河东解良人也。亡命奔涿郡。先主于乡里合徒众，而羽与张飞为之御侮。先主为平原相，以羽、飞为别部司马，分统部曲。先主与二人寝则同床，恩若兄弟。而稠人广坐，侍立终日，随先主周旋，不避艰险。先主之袭杀徐州刺史车胄，使关羽守下邳城，行太守事，而身还小沛。""",
        
        """张飞字益德，涿郡人也，少与关羽俱事先主。羽年长数岁，飞兄事之。先主从曹公破吕布于下邳，随还许都，曹公拜飞为中郎将。先主背曹公依袁绍、刘表。表卒，曹公入荆州，先主奔江南。曹公追之，一日一夜，及于当阳之长坂。先主闻曹公卒至，弃妻子走，使飞将二十骑拒后。飞据水断桥，瞋目横矛，曰：身是张益德也，可来共决死！声如巨雷，曹军皆惊。"""
    ]
    
    print("=== 主题建模方法对比演示 ===")
    print()
    
    # 测试LDA
    print("1. 使用LDA进行主题建模：")
    print("-" * 50)
    
    try:
        lda_extractor = MetadataExtractor(
            enable_topic_modeling=True,
            topic_model_type="lda",
            cache_enabled=False
        )
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n文本 {i} (LDA):")
            metadata = lda_extractor.extract_metadata(text)
            print(f"主题关键词: {metadata.topics}")
            print(f"关键词: {metadata.keywords[:5]}")
            print(f"人物实体: {metadata.entities.get('PERSON', [])[:5]}")
            
    except Exception as e:
        print(f"LDA测试失败: {e}")
    
    print("\n" + "=" * 60)
    
    # 测试BERTopic
    print("\n2. 使用BERTopic进行主题建模：")
    print("-" * 50)
    
    try:
        bertopic_extractor = MetadataExtractor(
            enable_topic_modeling=True,
            topic_model_type="bertopic",
            cache_enabled=False
        )
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n文本 {i} (BERTopic):")
            metadata = bertopic_extractor.extract_metadata(text)
            print(f"主题关键词: {metadata.topics}")
            print(f"关键词: {metadata.keywords[:5]}")
            print(f"人物实体: {metadata.entities.get('PERSON', [])[:5]}")
            
    except Exception as e:
        print(f"BERTopic测试失败: {e}")
    
    print("\n" + "=" * 60)
    
    # 测试关键词提取（回退方案）
    print("\n3. 仅使用关键词提取（回退方案）：")
    print("-" * 50)
    
    try:
        keyword_extractor = MetadataExtractor(
            enable_topic_modeling=False,  # 禁用主题建模
            cache_enabled=False
        )
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n文本 {i} (关键词):")
            metadata = keyword_extractor.extract_metadata(text)
            print(f"主题关键词: {metadata.topics}")
            print(f"关键词: {metadata.keywords[:5]}")
            print(f"人物实体: {metadata.entities.get('PERSON', [])[:5]}")
            
    except Exception as e:
        print(f"关键词提取测试失败: {e}")
    
    print("\n=== 对比总结 ===")
    print("1. LDA: 基于统计的主题建模，适合发现文档集合中的潜在主题")
    print("2. BERTopic: 基于BERT嵌入的主题建模，语义理解更好但需要更多文档")
    print("3. 关键词提取: 简单快速的回退方案，基于词频和jieba分词")
    print("\n建议：")
    print("- 文档数量少时使用LDA或关键词提取")
    print("- 文档数量多且需要高质量主题时使用BERTopic")
    print("- 对性能要求高时使用关键词提取")

if __name__ == "__main__":
    test_topic_models()