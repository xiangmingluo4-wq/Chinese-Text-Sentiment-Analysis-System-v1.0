# -*- coding: utf-8 -*-
"""
统一配置文件
中文文本情感分析系统 - 全局配置管理
（支持 XGBoost + 特征工程，轻量高效）
"""

import os

# ========== 路径配置 ==========
# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
# 模型目录
MODEL_DIR = os.path.join(BASE_DIR, 'models')
# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
# 文档目录
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
# 代码目录
CODE_DIR = os.path.join(BASE_DIR, 'code')
# 训练脚本目录
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
# 模板目录
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
# 静态文件目录
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# ========== 数据文件路径 ==========
# 情感分析数据集
SENTIMENT_DATA_PATH = os.path.join(DATA_DIR, 'sentiment_dataset_10000.csv')
# 问答知识库
QA_KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, 'qa_knowledge_base.csv')
# 未知问题回流文件
QA_UNKNOWN_QUESTIONS_PATH = os.path.join(DATA_DIR, 'qa_unknown_questions.csv')

# ========== 模型文件路径 ==========
# SVM模型（备用）
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
# TF-IDF向量化器
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
# XGBoost模型（主推荐，轻量 < 50MB）
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')

# ========== 模型参数 ==========
# TF-IDF最大特征数（可适当增加）
MAX_FEATURES = 8000
# N-gram范围（扩展为三元组以捕捉更多短语）
NGRAM_RANGE = (1, 3)
# 测试集比例
TEST_SIZE = 0.2
# 随机种子（保证可复现）
RANDOM_STATE = 42
# SVM核函数
SVM_KERNEL = 'linear'
# SVM正则化参数
SVM_C = 1.0
# 是否启用概率估计
SVM_PROBABILITY = True

# ========== XGBoost 配置（轻量级，推荐） ==========
# XGBoost 参数
XGBOOST_PARAMS = {
    'n_estimators': 200,           # 树的数量
    'max_depth': 6,                # 最大深度
    'learning_rate': 0.1,          # 学习率
    'subsample': 0.8,              # 样本采样
    'colsample_bytree': 0.8,       # 特征采样
    'eval_metric': 'mlogloss',     # 多分类损失
    'random_state': 42,
    'use_label_encoder': False
}
# 是否优先使用 XGBoost（若模型存在）
PREFER_XGBOOST = True

# ========== 特征工程配置 ==========
# 是否启用额外特征（情感词典、词性比例、文本长度等）
ENABLE_EXTRA_FEATURES = True
# 情感词典路径（可自定义扩充）
SENTIMENT_DICT_PATH = os.path.join(DATA_DIR, 'sentiment_dict.csv')

# ========== 质检阈值配置 ==========
# Kappa系数阈值（训练准入）
KAPPA_THRESHOLD = 0.7
# 异常样本率阈值（模板生成数据允许较高重复率）
ANOMALY_RATE_THRESHOLD = 0.5
# 文本最短长度（字符）
MIN_TEXT_LENGTH = 2
# 文本最长长度（字符）
MAX_TEXT_LENGTH = 200
# 抽样复检比例
SAMPLING_RATIO = 0.05

# ========== 情感映射配置 ==========
# 英文标签到中文显示的映射
SENTIMENT_MAP = {
    'positive': '正面😊',
    'negative': '负面😠',
    'neutral': '中性😐'
}

# 情感标签列表
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']

# 情感颜色映射（前端可视化用）
SENTIMENT_COLORS = {
    'positive': '#10b981',  # 绿色
    'negative': '#ef4444',  # 红色
    'neutral': '#6b7280'    # 灰色
}

# ========== 智能客服配置 ==========
# 问答相似度阈值
QA_SIMILARITY_THRESHOLD = 0.5
# 负面情绪置信度阈值（触发安抚）
NEGATIVE_CONFIDENCE_THRESHOLD = 0.85
# 知识性问题关键词
KNOWLEDGE_KEYWORDS = [
    '什么是', '什么叫', '如何', '怎么', '请问', '怎样',
    '为什么', '咋', '咋弄', '咋搞', '如何使用', '怎么用'
]

# 负面安抚话术库
NEGATIVE_COMFORT_MESSAGES = [
    '非常抱歉给您带来不好的体验，我来帮您解决问题。',
    '理解您的心情，我们会努力改进的。',
    '感谢您的反馈，这对我们很重要。',
    '很抱歉让您失望了，请问有什么可以帮您的吗？'
]

# ========== 领域配置 ==========
# 10大领域列表
DOMAINS = [
    '电商购物', '餐饮美食', '影视娱乐', '酒店住宿', '旅游出行',
    '教育培训', '数码产品', '美妆护肤', '图书阅读', '医疗健康'
]

# ========== 停用词配置 ==========
# 常用中文停用词
STOP_WORDS = [
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '或',
    '在', '也', '很', '有', '这', '那', '我', '你', '他', '她',
    '它', '们', '个', '一', '不', '人', '都', '上', '下', '中',
    '为', '以', '到', '说', '要', '去', '来', '着', '过', '得',
    '地', '啊', '呀', '吧', '呢', '吗', '哦', '哈', '嗯', '哎'
]

# ========== 深度学习配置（可选，保留但默认不启用） ==========
# BERT 模型名称或路径（若本地存在则使用本地，否则从HuggingFace下载）
BERT_MODEL_NAME = 'uer/roberta-base-finetuned-dianping-chinese'
# BERT 最大序列长度
BERT_MAX_LENGTH = 128
# 设备：None 表示自动检测 cuda/cpu，也可手动指定 'cuda' 或 'cpu'
BERT_DEVICE = None
# 是否启用 BERT（设为 False 则完全忽略，避免加载大模型）
ENABLE_BERT = False   # 关闭 BERT，使用轻量级 XGBoost

# ========== Flask服务配置 ==========
# 服务主机
FLASK_HOST = '0.0.0.0'
# 服务端口
FLASK_PORT = 5000
# 是否开启调试模式
FLASK_DEBUG = False
# 是否允许跨域
ENABLE_CORS = True

# ========== 日志配置 ==========
# 日志级别
LOG_LEVEL = 'INFO'
# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ========== 版本信息 ==========
# 项目版本（轻量级增强版）
PROJECT_VERSION = '2.1.0'
# 项目名称
PROJECT_NAME = '中文文本情感分析系统'

# ========== 持续反馈闭环配置 ==========
# 反馈数据存储路径
FEEDBACK_CSV_PATH = os.path.join(DATA_DIR, 'feedback_log.csv')
# 原始训练数据路径（用于混合重训，防止灾难性遗忘）
ORIGINAL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')  # 请放置您原始的训练集
# 自动重训触发阈值（每收集到 N 条新反馈，自动触发一次重训）
RETRAIN_THRESHOLD = 10
# 重训后模型保存路径（优先保存 XGBoost，若不可用则保存 SVM）
RETRAIN_MODEL_PATH = XGBOOST_MODEL_PATH   # 改为 XGBoost
RETRAIN_VEC_PATH = TFIDF_VECTORIZER_PATH


def init_dirs():
    """
    初始化必要的目录
    确保所有需要的目录都存在
    """
    dirs_to_create = [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        DOCS_DIR,
        os.path.join(DOCS_DIR, '质检报告'),
        CODE_DIR,
        TRAIN_DIR,
        TEMPLATES_DIR,
        os.path.join(STATIC_DIR, 'css'),
        os.path.join(STATIC_DIR, 'js'),
    ]

    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")


if __name__ == '__main__':
    # 测试配置
    print(f"项目名称: {PROJECT_NAME}")
    print(f"项目版本: {PROJECT_VERSION}")
    print(f"基础路径: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"XGBoost 模型路径: {XGBOOST_MODEL_PATH}")
    print(f"特征工程启用: {ENABLE_EXTRA_FEATURES}")
    print(f"BERT 启用: {ENABLE_BERT}")
    print(f"\n正在初始化目录...")
    init_dirs()
    print("目录初始化完成！")