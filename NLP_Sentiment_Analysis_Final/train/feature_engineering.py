# -*- coding: utf-8 -*-
"""
特征工程模块：提取额外特征用于模型训练（增强版）
新增特征：标点符号统计、否定词检测、情感极性得分、平均词长等
"""
import re
import jieba
import jieba.posseg as pseg
import pandas as pd
from collections import Counter

# ========== 扩充情感词典 ==========
POSITIVE_WORDS = {
    '好', '棒', '赞', '满意', '喜欢', '开心', '高兴', '优秀', '完美',
    '惊艳', '超值', '划算', '值得', '推荐', '好评', '给力', '强大',
    '实用', '舒适', '干净', '明亮', '周到', '热情', '专业', '细致',
    '温馨', '精致', '良心', '地道', '一流', '绝佳', '出色', '靠谱',
    '实惠', '实在', '划算', '划算', '喜欢', '满意', '赞不绝口', '好评如潮',
    '干净', '整洁', '宽敞', '明亮', '优雅', '高大上', '精致', '舒适',
    '安静', '悠闲', '惬意', '愉快', '感动', '温暖', '贴心', '耐心',
    '负责', '高效', '神速', '稳定', '流畅', '清晰', '细腻', '丰富',
    '有趣', '精彩', '振奋', '受益', '有用', '经典', '深刻', '真实'
}
NEGATIVE_WORDS = {
    '差', '烂', '垃圾', '糟糕', '恶心', '失望', '无语', '坑', '差劲',
    '不好', '不行', '没用', '浪费', '贵', '慢', '卡', '脏', '吵',
    '态度差', '不负责', '误导', '欺骗', '敷衍', '不耐烦', '冷漠', '恶劣',
    '坑爹', '渣', '土', '旧', '破', '难用', '无用', '无用功', '无效',
    '麻烦', '累赘', '繁琐', '难以忍受', '痛苦', '心烦', '焦虑', '不满',
    '投诉', '差评', '退货', '退款', '差劲', '糟糕', '烂透了', '恶心死了'
}

# ========== 否定词列表 ==========
NEGATIVE_WORDS_LIST = [
    '不', '没', '无', '非', '莫', '勿', '别', '未', '休', '毫不',
    '毫无', '没有', '不是', '不能', '不会', '不行', '不用', '不太'
]

# ========== 特征提取函数 ==========
def count_sentiment_words(text):
    """统计情感词数量"""
    pos_count = sum(1 for w in text if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in text if w in NEGATIVE_WORDS)
    return pos_count, neg_count

def extract_pos_ratio(text):
    """提取词性比例特征"""
    words = pseg.lcut(text)
    total = len(words)
    if total == 0:
        return 0.0, 0.0, 0.0
    adj_count = sum(1 for _, flag in words if flag.startswith('a'))  # 形容词
    verb_count = sum(1 for _, flag in words if flag.startswith('v'))  # 动词
    noun_count = sum(1 for _, flag in words if flag.startswith('n'))  # 名词
    return adj_count/total, verb_count/total, noun_count/total

def extract_punctuation_features(text):
    """提取标点符号特征"""
    exclamation = text.count('！') + text.count('!')
    question = text.count('？') + text.count('?')
    ellipsis = text.count('……') + text.count('...') + text.count('…')
    comma = text.count('，') + text.count(',')
    period = text.count('。') + text.count('.')
    return {
        'exclamation_cnt': exclamation,
        'question_cnt': question,
        'ellipsis_cnt': ellipsis,
        'comma_cnt': comma,
        'period_cnt': period
    }

def detect_negation(text):
    """检测是否包含否定词"""
    for neg in NEGATIVE_WORDS_LIST:
        if neg in text:
            return 1
    return 0

def extract_extra_features(text):
    """
    提取额外特征，返回一个字典
    特征包括：
    - 文本长度 (text_len)
    - 正面词计数 (pos_word_cnt)
    - 负面词计数 (neg_word_cnt)
    - 情感极性得分 (sentiment_score = pos_cnt - neg_cnt)
    - 形容词比例 (adj_ratio)
    - 动词比例 (verb_ratio)
    - 名词比例 (noun_ratio)
    - 平均词长 (avg_word_len)
    - 感叹号数量 (exclamation_cnt)
    - 问号数量 (question_cnt)
    - 省略号数量 (ellipsis_cnt)
    - 逗号数量 (comma_cnt)
    - 句号数量 (period_cnt)
    - 是否包含否定词 (has_negation)
    - 按标点分割的句子数量 (sentence_count)
    """
    if not text or len(text.strip()) == 0:
        return {
            'text_len': 0,
            'pos_word_cnt': 0,
            'neg_word_cnt': 0,
            'sentiment_score': 0,
            'adj_ratio': 0.0,
            'verb_ratio': 0.0,
            'noun_ratio': 0.0,
            'avg_word_len': 0.0,
            'exclamation_cnt': 0,
            'question_cnt': 0,
            'ellipsis_cnt': 0,
            'comma_cnt': 0,
            'period_cnt': 0,
            'has_negation': 0,
            'sentence_count': 1
        }

    # 文本长度
    text_len = len(text)

    # 情感词统计
    pos_cnt, neg_cnt = count_sentiment_words(text)
    sentiment_score = pos_cnt - neg_cnt

    # 词性比例（基于原文本分词）
    adj_ratio, verb_ratio, noun_ratio = extract_pos_ratio(text)

    # 平均词长（按分词结果计算）
    words = list(jieba.cut(text))
    total_char = sum(len(w) for w in words)
    avg_word_len = total_char / len(words) if words else 0.0

    # 标点特征
    punct = extract_punctuation_features(text)

    # 否定词
    has_neg = detect_negation(text)

    # 句子数量（按句号、问号、感叹号分割）
    sentence_count = len(re.split(r'[。！？!?]', text)) - 1
    if sentence_count == 0:
        sentence_count = 1

    return {
        'text_len': text_len,
        'pos_word_cnt': pos_cnt,
        'neg_word_cnt': neg_cnt,
        'sentiment_score': sentiment_score,
        'adj_ratio': adj_ratio,
        'verb_ratio': verb_ratio,
        'noun_ratio': noun_ratio,
        'avg_word_len': avg_word_len,
        'exclamation_cnt': punct['exclamation_cnt'],
        'question_cnt': punct['question_cnt'],
        'ellipsis_cnt': punct['ellipsis_cnt'],
        'comma_cnt': punct['comma_cnt'],
        'period_cnt': punct['period_cnt'],
        'has_negation': has_neg,
        'sentence_count': sentence_count
    }

def transform_extra_features(texts):
    """批量提取额外特征，返回 DataFrame"""
    records = [extract_extra_features(t) for t in texts]
    return pd.DataFrame(records)

# ========== 测试（可选） ==========
if __name__ == '__main__':
    sample_texts = [
        "这个手机真的太棒了，拍照效果非常好！",
        "上菜太慢，等很久，服务态度也差。",
        "包装完好，正常使用。",
        "果然没让我失望，太棒了！",
        "常规购买。"
    ]
    df = transform_extra_features(sample_texts)
    print("提取的特征示例：")
    print(df.head())
    print("\n特征列名：", df.columns.tolist())

