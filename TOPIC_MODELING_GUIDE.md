# 主题建模方法切换指南

本指南介绍如何在SemanticRAG系统中切换不同的主题建模方法。

## 支持的主题建模方法

### 1. LDA (Latent Dirichlet Allocation)
- **优点**: 经典的主题建模算法，计算效率高，适合中小规模文档集合
- **缺点**: 基于词袋模型，无法捕捉词序和语义信息
- **适用场景**: 文档数量较少，对性能要求较高的场景
- **依赖**: scikit-learn

### 2. BERTopic
- **优点**: 基于BERT嵌入，语义理解能力强，主题质量高
- **缺点**: 计算资源消耗大，需要较多文档才能正常工作
- **适用场景**: 文档数量多，对主题质量要求高的场景
- **依赖**: bertopic, sentence-transformers

### 3. 关键词提取 (回退方案)
- **优点**: 速度快，资源消耗少，稳定可靠
- **缺点**: 不是真正的主题建模，只是关键词提取
- **适用场景**: 作为其他方法失败时的回退方案
- **依赖**: jieba

## 如何切换主题建模方法

### 方法1: 修改主程序

在 `rag_system/main.py` 中修改 `MetadataExtractor` 的初始化参数：

```python
# 使用LDA
metadata_extractor = MetadataExtractor(
    enable_topic_modeling=True,
    topic_model_type="lda"  # 选择LDA
)

# 使用BERTopic
metadata_extractor = MetadataExtractor(
    enable_topic_modeling=True,
    topic_model_type="bertopic"  # 选择BERTopic
)

# 禁用主题建模，仅使用关键词提取
metadata_extractor = MetadataExtractor(
    enable_topic_modeling=False  # 禁用主题建模
)
```

### 方法2: 通过配置文件

可以在 `utils/config.py` 中添加配置选项：

```python
# 主题建模配置
TOPIC_MODEL_TYPE = "lda"  # "lda", "bertopic", 或 "none"
ENABLE_TOPIC_MODELING = True
```

## 性能对比

| 方法 | 速度 | 内存消耗 | 主题质量 | 最小文档数 |
|------|------|----------|----------|------------|
| LDA | 快 | 低 | 中等 | 3-5个句子 |
| BERTopic | 慢 | 高 | 高 | 3个以上段落 |
| 关键词提取 | 很快 | 很低 | 低 | 1个文档 |

## 安装依赖

### LDA方法
```bash
pip install scikit-learn
```

### BERTopic方法
```bash
pip install bertopic sentence-transformers
```

### 关键词提取
```bash
pip install jieba
```

## 使用建议

1. **开发和测试阶段**: 使用LDA或关键词提取，速度快，便于调试
2. **生产环境**: 根据文档数量和质量要求选择
   - 文档少(<100个): LDA
   - 文档多(>1000个): BERTopic
   - 性能优先: 关键词提取
3. **混合策略**: 可以根据文档长度动态选择方法

## 故障排除

### LDA相关问题
- 如果出现"sklearn not available"错误，安装scikit-learn
- 如果主题质量差，尝试调整`n_components`参数

### BERTopic相关问题
- 如果出现"Transform unavailable"错误，说明文档数量不足
- 如果内存不足，考虑使用LDA替代

### 通用问题
- 如果所有主题建模方法都失败，系统会自动回退到关键词提取
- 检查jieba是否正确安装和初始化

## 示例代码

运行对比测试：
```bash
python topic_model_comparison.py
```

这个脚本会展示三种方法在相同文本上的效果对比。