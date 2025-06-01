"""元数据提取器模块，用于从文本中提取主题、实体等丰富的元数据信息。

该模块提供以下功能：
1. 主题建模（使用BERTopic或简单的关键词提取）
2. 命名实体识别（人物、地点、组织等）
3. 位置和结构信息提取
4. 缓存机制以提高性能
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
import re
import hashlib
import json
from collections import Counter
from dataclasses import dataclass

# 可选依赖导入
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Using simple keyword extraction.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Using rule-based NER.")

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: BERTopic not available. Using keyword-based topic extraction.")

try:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False
    print("Warning: scikit-learn not available for LDA. Install with: pip install scikit-learn")


@dataclass
class ExtractedMetadata:
    """提取的元数据结构"""
    topics: List[str]  # 主题关键词
    entities: Dict[str, List[str]]  # 实体：{"PERSON": [...], "LOC": [...], "ORG": [...]}
    keywords: List[str]  # 关键词
    sentiment: Optional[str] = None  # 情感（可选）
    complexity_score: Optional[float] = None  # 复杂度评分（可选）


class MetadataExtractor:
    """元数据提取器主类"""
    
    def __init__(self, 
                 enable_topic_modeling: bool = True,
                 enable_ner: bool = True,
                 enable_keyword_extraction: bool = True,
                 cache_enabled: bool = True,
                 topic_model_type: str = "lda"):
        """
        初始化元数据提取器
        
        Args:
            enable_topic_modeling: 是否启用主题建模
            enable_ner: 是否启用命名实体识别
            enable_keyword_extraction: 是否启用关键词提取
            cache_enabled: 是否启用缓存
            topic_model_type: 主题模型类型 ("bertopic" 或 "lda")
        """
        self.enable_topic_modeling = enable_topic_modeling
        self.enable_ner = enable_ner
        self.enable_keyword_extraction = enable_keyword_extraction
        self.cache_enabled = cache_enabled
        self.topic_model_type = topic_model_type.lower()
        
        # 初始化缓存
        self._cache: Dict[str, ExtractedMetadata] = {}
        
        # 初始化NLP工具
        self._init_nlp_tools()
        
        # 三国演义特定的实体词典
        self._init_sanguo_entities()
    
    def _init_nlp_tools(self):
        """初始化NLP工具"""
        global SPACY_AVAILABLE, BERTOPIC_AVAILABLE
        
        # 初始化spaCy
        self.nlp = None
        if SPACY_AVAILABLE and self.enable_ner:
            try:
                self.nlp = spacy.load("zh_core_web_sm")
            except OSError:
                print("Warning: Chinese spaCy model not found. Install with: python -m spacy download zh_core_web_sm")
                SPACY_AVAILABLE = False
        
        # 初始化主题模型
        self.topic_model = None
        self.lda_model = None
        self.vectorizer = None
        
        if self.enable_topic_modeling:
            if self.topic_model_type == "bertopic" and BERTOPIC_AVAILABLE:
                try:
                    # 使用中文预训练模型
                    self.topic_model = BERTopic(
                        language="chinese (simplified)",
                        calculate_probabilities=True,
                        verbose=False
                    )
                    print("Using BERTopic for topic modeling")
                except Exception as e:
                    print(f"Warning: Failed to initialize BERTopic: {e}")
                    self.topic_model = None
            elif self.topic_model_type == "lda" and LDA_AVAILABLE:
                try:
                    # 初始化LDA模型和向量化器
                    self.vectorizer = CountVectorizer(
                        max_features=1000,
                        stop_words=None,  # 我们会手动处理中文停用词
                        token_pattern=r'[\u4e00-\u9fff]+',  # 只匹配中文字符
                        min_df=1,
                        max_df=0.95
                    )
                    self.lda_model = LatentDirichletAllocation(
                        n_components=5,  # 主题数量
                        random_state=42,
                        max_iter=10,
                        learning_method='batch'
                    )
                    print("Using LDA for topic modeling")
                except Exception as e:
                    print(f"Warning: Failed to initialize LDA: {e}")
                    self.lda_model = None
            else:
                print(f"Warning: Topic model '{self.topic_model_type}' not available or not supported")
        
        # 初始化jieba
        if JIEBA_AVAILABLE:
            jieba.initialize()
    
    def _init_sanguo_entities(self):
        """初始化三国演义特定的实体词典"""
        # 主要人物
        self.sanguo_persons = {
            "刘备", "关羽", "张飞", "诸葛亮", "赵云", "马超", "黄忠", "魏延", "姜维",
            "曹操", "曹丕", "曹植", "司马懿", "司马昭", "司马师", "夏侯惇", "夏侯渊", "张辽", "徐晃", "张郃", "于禁", "乐进", "李典",
            "孙权", "孙策", "孙坚", "周瑜", "鲁肃", "吕蒙", "陆逊", "太史慈", "甘宁", "黄盖", "程普", "韩当", "蒋钦", "周泰",
            "董卓", "吕布", "袁绍", "袁术", "刘表", "刘璋", "马腾", "韩遂", "公孙瓒", "陶谦",
            "貂蝉", "大乔", "小乔", "甄氏", "糜夫人", "孙夫人"
        }
        
        # 主要地点
        self.sanguo_locations = {
            "洛阳", "长安", "许昌", "邺城", "成都", "建业", "江陵", "襄阳", "新野", "汝南",
            "赤壁", "夷陵", "街亭", "五丈原", "剑阁", "汉中", "定军山", "麦城", "白帝城",
            "虎牢关", "潼关", "函谷关", "剑门关", "阳平关", "华容道", "长坂坡", "博望坡",
            "徐州", "青州", "兖州", "冀州", "并州", "凉州", "益州", "荆州", "扬州", "交州"
        }
        
        # 主要组织/势力
        self.sanguo_organizations = {
            "蜀汉", "曹魏", "东吴", "黄巾军", "西凉军", "并州军", "青州兵", "荆州军",
            "五虎上将", "五子良将", "江东十二虎臣", "建安七子", "竹林七贤"
        }
        
        # 如果jieba可用，添加自定义词典
        if JIEBA_AVAILABLE:
            for person in self.sanguo_persons:
                jieba.add_word(person, freq=1000, tag='nr')  # nr: 人名
            for location in self.sanguo_locations:
                jieba.add_word(location, freq=800, tag='ns')  # ns: 地名
            for org in self.sanguo_organizations:
                jieba.add_word(org, freq=600, tag='nt')  # nt: 机构名
    
    def extract_metadata(self, text: str, chapter_info: Optional[Dict[str, Any]] = None) -> ExtractedMetadata:
        """
        从文本中提取元数据
        
        Args:
            text: 输入文本
            chapter_info: 章节信息（可选）
            
        Returns:
            提取的元数据
        """
        # 检查缓存
        if self.cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # 提取各类元数据
        topics = self._extract_topics(text) if self.enable_topic_modeling else []
        entities = self._extract_entities(text) if self.enable_ner else {}
        keywords = self._extract_keywords(text) if self.enable_keyword_extraction else []
        
        # 创建元数据对象
        metadata = ExtractedMetadata(
            topics=topics,
            entities=entities,
            keywords=keywords
        )
        
        # 缓存结果
        if self.cache_enabled:
            self._cache[cache_key] = metadata
        
        return metadata
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _extract_topics(self, text: str) -> List[str]:
        """提取主题关键词"""
        if len(text) < 100:
            # 文本太短，直接使用关键词提取
            return self._extract_keywords(text)[:3]
        
        # 根据配置的模型类型进行主题提取
        if self.topic_model_type == "bertopic" and self.topic_model:
            return self._extract_topics_bertopic(text)
        elif self.topic_model_type == "lda" and self.lda_model:
            return self._extract_topics_lda(text)
        
        # 回退到关键词提取
        return self._extract_keywords(text)[:3]
    
    def _extract_topics_bertopic(self, text: str) -> List[str]:
        """使用BERTopic提取主题"""
        try:
            # 将长文本分割成多个段落来模拟多文档
            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
            
            # 如果段落数量足够，使用BERTopic
            if len(paragraphs) >= 3:
                topics, _ = self.topic_model.fit_transform(paragraphs)
                topic_info = self.topic_model.get_topic_info()
                
                # 获取主要主题的关键词
                if len(topic_info) > 1:  # 排除噪声主题(-1)
                    main_topic = topic_info.iloc[1]['Topic']  # 第一个真实主题
                    topic_words = self.topic_model.get_topic(main_topic)
                    return [word for word, _ in topic_words[:5]]  # 取前5个关键词
        except Exception as e:
            print(f"BERTopic extraction failed: {e}")
        
        return self._extract_keywords(text)[:3]
    
    def _extract_topics_lda(self, text: str) -> List[str]:
        """使用LDA提取主题"""
        try:
            # 将长文本分割成句子或段落
            sentences = [s.strip() for s in re.split(r'[。！？\n]', text) if len(s.strip()) > 10]
            
            if len(sentences) < 3:
                # 句子太少，回退到关键词提取
                return self._extract_keywords(text)[:3]
            
            # 预处理文本：移除停用词
            processed_sentences = []
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '与', '或', '但', '而', '因为', '所以', '如果', '虽然', '然而', '因此', '于是', '然后', '接着', '最后', '首先', '其次', '再次', '另外', '此外', '总之', '综上'}
            
            for sentence in sentences:
                # 使用jieba分词
                if JIEBA_AVAILABLE:
                    words = jieba.lcut(sentence)
                else:
                    words = re.findall(r'[\u4e00-\u9fff]+', sentence)
                
                # 过滤停用词和短词
                filtered_words = [w for w in words if len(w) >= 2 and w not in stop_words]
                if filtered_words:
                    processed_sentences.append(' '.join(filtered_words))
            
            if len(processed_sentences) < 2:
                return self._extract_keywords(text)[:3]
            
            # 向量化
            doc_term_matrix = self.vectorizer.fit_transform(processed_sentences)
            
            # LDA主题建模
            self.lda_model.fit(doc_term_matrix)
            
            # 获取主题词
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            # 获取最重要的主题
            topic_idx = 0  # 使用第一个主题
            top_words_idx = self.lda_model.components_[topic_idx].argsort()[-5:][::-1]
            topics = [feature_names[i] for i in top_words_idx]
            
            return topics[:3] if topics else self._extract_keywords(text)[:3]
            
        except Exception as e:
            print(f"LDA extraction failed: {e}")
            return self._extract_keywords(text)[:3]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        if JIEBA_AVAILABLE:
            try:
                # 使用jieba的TF-IDF关键词提取
                keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
                return keywords
            except Exception as e:
                print(f"Jieba keyword extraction failed: {e}")
        
        # 简单的关键词提取（回退方案）
        return self._simple_keyword_extraction(text)
    
    def _simple_keyword_extraction(self, text: str) -> List[str]:
        """简单的关键词提取（回退方案）"""
        # 移除标点符号，分词
        words = re.findall(r'[\u4e00-\u9fff]+', text)  # 只保留中文字符
        
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        filtered_words = [w for w in words if len(w) >= 2 and w not in stop_words]
        
        # 统计词频
        word_counts = Counter(filtered_words)
        
        # 返回最常见的词
        return [word for word, _ in word_counts.most_common(10)]
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取命名实体"""
        entities = {"PERSON": [], "LOC": [], "ORG": [], "EVENT": []}
        
        # 使用spaCy进行NER
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        entities["PERSON"].append(ent.text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities["LOC"].append(ent.text)
                    elif ent.label_ == "ORG":
                        entities["ORG"].append(ent.text)
            except Exception as e:
                print(f"spaCy NER failed: {e}")
        
        # 使用三国演义特定词典进行补充识别
        entities = self._extract_sanguo_entities(text, entities)
        
        # 去重并限制数量
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]  # 每类最多10个
        
        return entities
    
    def _extract_sanguo_entities(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """使用三国演义特定词典提取实体"""
        # 提取人物
        for person in self.sanguo_persons:
            if person in text:
                entities["PERSON"].append(person)
        
        # 提取地点
        for location in self.sanguo_locations:
            if location in text:
                entities["LOC"].append(location)
        
        # 提取组织
        for org in self.sanguo_organizations:
            if org in text:
                entities["ORG"].append(org)
        
        # 提取事件（基于关键词模式）
        event_patterns = [
            r'(\w*战\w*)', r'(\w*之战)', r'(\w*会\w*)', r'(\w*之盟)',
            r'(起兵\w*)', r'(讨伐\w*)', r'(征\w*)', r'(伐\w*)'
        ]
        
        for pattern in event_patterns:
            matches = re.findall(pattern, text)
            entities["EVENT"].extend(matches)
        
        return entities
    
    def enrich_chapter_metadata(self, chapter_chunks: List[Any]) -> List[Any]:
        """
        为章节分块添加丰富的元数据
        
        Args:
            chapter_chunks: 章节分块列表
            
        Returns:
            增强了元数据的章节分块列表
        """
        enriched_chunks = []
        
        for chunk in chapter_chunks:
            # 提取元数据
            extracted_meta = self.extract_metadata(chunk.content)
            
            # 合并到现有元数据中
            if chunk.metadata is None:
                chunk.metadata = {}
            
            chunk.metadata.update({
                "topics": extracted_meta.topics,
                "entities": extracted_meta.entities,
                "keywords": extracted_meta.keywords,
                "entity_count": {k: len(v) for k, v in extracted_meta.entities.items()},
                "keyword_count": len(extracted_meta.keywords),
                "content_length": len(chunk.content),
                "has_dialogue": self._detect_dialogue(chunk.content),
                "has_battle": self._detect_battle_content(chunk.content)
            })
            
            enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def _detect_dialogue(self, text: str) -> bool:
        """检测文本中是否包含对话"""
        dialogue_patterns = [r'"[^"]+"', r'"[^"]+"', r'曰：', r'道：', r'说道：']
        for pattern in dialogue_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _detect_battle_content(self, text: str) -> bool:
        """检测文本中是否包含战斗内容"""
        battle_keywords = ['战', '斗', '杀', '败', '胜', '攻', '守', '围', '破', '兵', '马', '刀', '剑', '弓', '箭']
        for keyword in battle_keywords:
            if keyword in text:
                return True
        return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.cache_enabled
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()