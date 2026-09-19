# -*- coding: utf-8 -*-
# 警告过滤必须放在所有import最前面
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Building prefix dict")
warnings.filterwarnings("ignore", message="Loading model")
warnings.filterwarnings("ignore", message="Prefix dict has been built")

"""
文本预处理脚本 V2.4 - 轻量级增强版
优化：保留否定词（对情感极性至关重要），调整停用词，适配 XGBoost 特征工程
"""

import os
import re
import jieba
import jieba.posseg as pseg
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ==================== 全局配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 向量化参数（适配 XGBoost 训练）
MAX_FEATURES = 8000
NGRAM_RANGE = (1, 3)          # 扩展为三元组以捕获短语
MIN_DF = 3
MAX_DF = 0.9

# 词性筛选白名单（可选，默认不使用词性过滤）
ALLOW_POS = {
    'n', 'nr', 'ns', 'nt', 'nz',
    'v', 'vd', 'vn',
    'a', 'ad', 'an',
    'd', 'z',
    'i', 'l',
    'x'
}

# 自定义领域情感词典
CUSTOM_WORDS = [
    '性价比', '物超所值', '货不对板', '中规中矩', '差强人意',
    '踩雷', '避雷', '劝退', '智商税', '闭眼入', '绝绝子', 'yyds', '大踩雷',
    '续航', '发热', '卡顿', '丝滑', '像素', '音质', '充电',
    '上菜', '分量', '食材', '口味', '摆盘', '卫生',
    '隔音', '早餐', '民宿', '景区', '门票', '出片',
    '保湿', '闷痘', '脱妆', '服帖', '肤感', '拔干',
    '太棒了', '太差了', '很差', '很好', '超快', '超棒'
]

# ========== 停用词表（保留否定词，避免丢失情感反转信息） ==========
STOP_WORDS = {
    # 结构助词/介词/连词（无情感倾向）
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '也', '还',
    '在', '对', '为', '以', '把', '被', '给', '让', '到', '从', '向', '往',
    # 疑问/语气助词（保留“吗”“呢”等可能暗示反问语气，但为减少噪声仅保留少数）
    '吗', '呢', '吧', '啊', '呀', '哇', '唉', '哦', '呐',
    # 指代词（无情感）
    '这', '那', '哪', '什么', '怎么', '怎样', '为什么',
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '自己',
    # 量词
    '个', '只', '件', '条', '款', '种', '类', '份', '次', '下',
    # 纯数字
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    # 情态动词（可能影响语气但一般不带情感倾向）
    '可以', '能', '会', '要', '想', '觉得', '感觉', '认为',
    # 指代短语
    '这个', '那个', '这种', '那种', '这么', '那么',
    '就是', '只是', '还是', '但是', '不过', '所以', '因此',
    '然后', '接着', '首先', '其次', '最后', '总之',
    # 无意义通用名词
    '东西', '宝贝', '玩意',
    # 口语语气词（无情感倾向）
    '真的', '简直', '其实', '反正', '居然', '竟然', '果然',
    # 标点符号（会被清洗掉，这里保留以防万一）
    '，', '。', '！', '？', '、', '；', '：', '"', "'", '（', '）', '《', '》', '【', '】'
}
# ★★★ 关键调整：移除所有否定词，因为它们是情感反转的关键信号 ★★★
# 否定词列表（保留在文本中，不被过滤）
NEGATION_WORDS = {'不', '没', '无', '非', '莫', '勿', '别', '未', '休', '毫不', '毫无', '没有', '不是', '不能', '不会', '不行', '不用', '不太'}
# 从 STOP_WORDS 中删除否定词（若存在）
STOP_WORDS -= NEGATION_WORDS

# 初始化jieba
for word in CUSTOM_WORDS:
    jieba.add_word(word)
jieba.setLogLevel(jieba.logging.WARNING)  # 只输出警告及以上，屏蔽info日志


# ==================== 文本清洗模块 ====================
def fullwidth_to_halfwidth(text):
    """全角字符转半角"""
    result = []
    for char in text:
        code = ord(char)
        if code == 12288:
            code = 32
        elif 65281 <= code <= 65374:
            code -= 65248
        result.append(chr(code))
    return ''.join(result)


def remove_redundant_chars(text):
    """压缩重复字符，保留语义的同时减少冗余特征"""
    text = re.sub(r'([，。！？、~])\1+', r'\1', text)
    text = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', text)
    return text


def clean_text(text):
    """完整文本清洗流水线"""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    # 1. 全角转半角
    text = fullwidth_to_halfwidth(text)

    # 2. 去除网络噪声
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\S+#', '', text)
    text = re.sub(r'\w+@\w+\.\w+', '', text)

    # 3. 去除emoji
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 4. 压缩重复
    text = remove_redundant_chars(text)

    # 5. 字符白名单过滤
    text = re.sub(
        r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；："\'（）《》【】]',
        '', text
    )

    # 6. 空格归一化
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==================== 分词与停用词模块 ====================
def tokenize(text, use_pos_filter=False):
    """中文分词，可选词性筛选"""
    if not isinstance(text, str) or not text.strip():
        return []

    if use_pos_filter:
        words = []
        for word, flag in pseg.lcut(text):
            if flag[0] in ALLOW_POS or word in CUSTOM_WORDS:
                words.append(word)
        return words
    else:
        return jieba.lcut(text)


def remove_stopwords(words):
    """去除停用词（但不移除否定词）"""
    if not words:
        return []

    return [
        w for w in words
        if w not in STOP_WORDS and len(w.strip()) > 0 and not w.isspace()
    ]


# ==================== 核心预处理流程 ====================
def preprocess_single(text, use_pos_filter=False):
    """单条文本完整预处理（保留否定词）"""
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    words = tokenize(cleaned, use_pos_filter=use_pos_filter)
    if not words:
        return ""

    filtered = remove_stopwords(words)
    if not filtered:
        # 若全部被过滤，则保留原始分词结果（防止空字符串）
        return ' '.join(words)

    return ' '.join(filtered)


def preprocess_batch(texts, use_pos_filter=False):
    """批量文本预处理"""
    return [preprocess_single(t, use_pos_filter=use_pos_filter) for t in texts]


# ==================== TF-IDF 向量化模块 ====================
def build_tfidf_vectorizer(texts, max_features=MAX_FEATURES,
                           ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_df=MAX_DF):
    """构建TF-IDF向量化器（适配 XGBoost）"""
    print(f"[信息] 正在构建TF-IDF向量化器")
    print(f"       最大特征数: {max_features}")
    print(f"       Ngram范围: {ngram_range}")
    print(f"       词频过滤: min_df={min_df}, max_df={max_df}")

    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        raise ValueError("所有文本均为空，无法构建向量化器")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        smooth_idf=True,
        norm='l2'
    )

    vectorizer.fit(valid_texts)
    feature_count = len(vectorizer.get_feature_names_out())
    print(f"[信息] 向量化器构建完成，实际特征维度: {feature_count}")

    return vectorizer


def vectorize_texts(vectorizer, texts):
    """文本向量化"""
    return vectorizer.transform(texts)


def preprocess_dataframe(df, text_column='text', use_pos_filter=False):
    """DataFrame批量预处理 + 数据质量统计"""
    print(f"[信息] 正在预处理 {len(df)} 条文本...")

    df = df.copy()
    df['processed_text'] = df[text_column].apply(
        lambda x: preprocess_single(x, use_pos_filter=use_pos_filter)
    )

    # 数据质量统计
    empty_count = (df['processed_text'] == '').sum()
    avg_len = df['processed_text'].str.split().str.len().mean()

    if empty_count > 0:
        print(f"[警告] 有 {empty_count} 条文本预处理后为空")
    print(f"[信息] 平均文本长度: {avg_len:.1f} 词")
    print("[信息] 文本预处理完成")

    return df


# ==================== 分析辅助工具 ====================
def get_top_keywords(vectorizer, text, top_n=5):
    """
    基于TF-IDF提取文本权重最高的关键词（兼容 joblib 加载的 vectorizer）
    """
    # 必须使用和训练时完全一致的预处理逻辑
    processed_text = preprocess_single(text)
    if not processed_text.strip():
        return []

    # 文本向量化
    tfidf_vec = vectorizer.transform([processed_text])
    # 获取向量化器的全部特征词表
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        # 旧版 sklearn 兼容
        feature_names = vectorizer.get_feature_names()
    # 提取当前文本的词权重数组
    weights = tfidf_vec.toarray()[0]

    # 组装「词-权重」并按权重降序排序
    word_weight_pairs = list(zip(feature_names, weights))
    word_weight_pairs.sort(key=lambda x: x[1], reverse=True)

    # 过滤权重为0的词，取前top_n个
    keywords = [word for word, weight in word_weight_pairs if weight > 0][:top_n]

    # 兜底：TF-IDF无结果时，返回分词后长度>1的非停用词
    if not keywords:
        word_list = processed_text.split()
        keywords = list(set([w for w in word_list if len(w) > 1]))[:top_n]

    return keywords


def get_high_frequency_words(texts, top_n=20):
    """统计高频词"""
    all_words = []
    for text in texts:
        if text:
            all_words.extend(text.split())

    counter = Counter(all_words)
    return counter.most_common(top_n)


# ==================== 测试主程序 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("  中文文本情感分析系统 - 文本预处理 V2.4（保留否定词）")
    print("=" * 60)
    print()

    # 1. 单条核心测试
    test_text = "这个手机真的太太太太棒了！！！拍照效果绝绝子，性价比超高，闭眼入～@客服"
    print(f"测试文本: {test_text}")
    print(f"处理结果: {preprocess_single(test_text)}")
    print()

    # 2. 否定词专项测试（验证保留否定词）
    neg_text = "客服态度很差，非常不满意，劝退了。"
    print(f"否定词测试: {neg_text}")
    print(f"处理结果: {preprocess_single(neg_text)}")
    # 预期输出应包含"不"、"没"等否定词
    print()

    # 3. 批量测试
    test_texts = [
        "这个产品质量很好，非常满意！",
        "这个东西太差了，简直智商税，大踩雷！",
        "一般般吧，中规中矩，没有特别好也没有特别差。",
        "物流很快，包装也很精美，物超所值！",
        "客服态度很差，非常不满意，劝退了。"
    ]

    print(f"批量测试 {len(test_texts)} 条文本...")
    processed_texts = preprocess_batch(test_texts)

    print("\n预处理结果:")
    for i, (original, processed) in enumerate(zip(test_texts, processed_texts)):
        print(f"  [{i+1}] 原文: {original}")
        print(f"      处理: {processed}")
        print()

    # 4. TF-IDF 测试
    print("构建TF-IDF向量化器...")
    vectorizer = build_tfidf_vectorizer(processed_texts, max_features=100, min_df=1)

    print("\n高频词Top10:")
    for word, count in get_high_frequency_words(processed_texts, top_n=10):
        print(f"  {word}: {count}次")

    print("\n各文本Top5关键词:")
    for i, processed in enumerate(processed_texts):
        keywords = get_top_keywords(vectorizer, processed, top_n=5)
        print(f"  [{i+1}] {keywords}")

    print()
    print("[完成] 所有测试通过，预处理模块运行正常！")

