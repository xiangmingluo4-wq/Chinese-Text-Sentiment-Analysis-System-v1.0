# -*- coding: utf-8 -*-
"""
训练流水线 V2.0 - 支持 XGBoost（主） + SVM（备用）
集成特征工程、轻量级模型、自动降级
"""

import os
import sys
import time
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 尝试导入项目模块 ==========
# 确保 train 目录在路径中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

# 导入数据生成和预处理
from data_generator import generate_sentiment_data
from preprocessing import preprocess_dataframe, build_tfidf_vectorizer, vectorize_texts

# 导入特征工程（若存在）
try:
    from feature_engineering import transform_extra_features
    FEATURE_ENG_AVAILABLE = True
except ImportError:
    FEATURE_ENG_AVAILABLE = False
    print("[提示] feature_engineering 模块未找到，将不使用额外特征")

# 尝试导入 XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[提示] xgboost 未安装，将使用 SVM 作为主模型")

# 尝试导入 config.py 以使用统一参数
try:
    from config import *
    print("[信息] 使用 config.py 中的参数")
except ImportError:
    # 若 config 不存在，使用内置默认值
    print("[信息] 未找到 config.py，使用内置默认参数")
    MAX_FEATURES = 8000
    NGRAM_RANGE = (1, 3)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SVM_KERNEL = 'linear'
    SVM_C = 1.0
    SVM_PROBABILITY = True
    XGBOOST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'use_label_encoder': False
    }
    ENABLE_EXTRA_FEATURES = True

# 模型保存路径
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
VECTORIZER_SAVE_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# ==================== 训练参数（可在此调整） ====================
DATA_COUNT = 50000          # 生成样本数
TEST_SIZE = TEST_SIZE       # 测试集比例
RANDOM_STATE = RANDOM_STATE

# 是否优先使用 XGBoost
PREFER_XGBOOST = True if XGB_AVAILABLE else False

# ==========================================================

def main():
    print("=" * 65)
    print("  中文文本情感分析系统 - 训练流水线 V2.0")
    if PREFER_XGBOOST:
        print("  主模型：XGBoost（轻量级）")
    else:
        print("  主模型：SVM（备用）")
    print("=" * 65)
    total_start = time.time()

    # -------------------- 1. 生成数据集 --------------------
    print("\n[步骤 1/7] 生成样本数据...")
    df = generate_sentiment_data(total_count=DATA_COUNT)
    print(f"✅ 生成完成，共 {len(df)} 条数据")
    print("情感类别分布：")
    print(df['sentiment'].value_counts().sort_index().to_string())

    # -------------------- 2. 文本预处理 --------------------
    print("\n[步骤 2/7] 执行文本预处理...")
    df = preprocess_dataframe(df, text_column='text')
    df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)
    print(f"✅ 预处理完成，剩余有效样本 {len(df)} 条")

    # -------------------- 3. 划分训练/测试集 --------------------
    print("\n[步骤 3/7] 划分训练集与测试集...")
    X_raw = df['text'].tolist()          # 原始文本（用于额外特征）
    X_proc = df['processed_text'].tolist()  # 预处理后的文本
    y = df['sentiment'].tolist()

    X_raw_train, X_raw_test, X_proc_train, X_proc_test, y_train, y_test = train_test_split(
        X_raw, X_proc, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"训练集：{len(X_raw_train)} 条 | 测试集：{len(X_raw_test)} 条")

    # -------------------- 4. TF-IDF 向量化 --------------------
    print("\n[步骤 4/7] 构建 TF-IDF 向量化器...")
    vectorizer = build_tfidf_vectorizer(
        X_proc_train,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE
    )
    X_train_tfidf = vectorizer.transform(X_proc_train)
    X_test_tfidf = vectorizer.transform(X_proc_test)
    print(f"✅ TF-IDF 特征维度：{X_train_tfidf.shape[1]}")

    # -------------------- 5. 额外特征提取（若可用） --------------------
    if FEATURE_ENG_AVAILABLE and ENABLE_EXTRA_FEATURES:
        print("\n[步骤 5/7] 提取额外特征...")
        train_extra = transform_extra_features(X_raw_train)
        test_extra = transform_extra_features(X_raw_test)
        X_train_extra = train_extra.values
        X_test_extra = test_extra.values
        # 合并特征
        X_train = hstack([X_train_tfidf, X_train_extra])
        X_test = hstack([X_test_tfidf, X_test_extra])
        print(f"✅ 额外特征维度：{X_train_extra.shape[1]}，合并后总维度：{X_train.shape[1]}")
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf
        print("[信息] 未启用额外特征")

    # -------------------- 6. 训练模型 --------------------
    print("\n[步骤 6/7] 训练模型...")
    train_start = time.time()

    # 标签编码（适用于 XGBoost）
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_
    print(f"类别映射：{dict(zip(classes, range(len(classes))))}")

    # 选择模型
    if PREFER_XGBOOST and XGB_AVAILABLE:
        print("使用 XGBoost 训练...")
        # 准备 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_enc)
        dtest = xgb.DMatrix(X_test, label=y_test_enc)

        # 复制参数并设置 num_class
        params = XGBOOST_PARAMS.copy()
        params['num_class'] = len(classes)
        params['objective'] = 'multi:softprob'

        # 训练
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 200),
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        # 保存模型和编码器
        joblib.dump(model, XGBOOST_MODEL_PATH)
        joblib.dump(le, LABEL_ENCODER_PATH)
        print(f"✅ XGBoost 模型已保存至 {XGBOOST_MODEL_PATH}")

        # 预测
        y_pred_proba = model.predict(dtest)
        y_pred_enc = np.argmax(y_pred_proba, axis=1)
        y_pred = le.inverse_transform(y_pred_enc)
        model_type = "XGBoost"

    else:
        # 降级到 SVM
        print("使用 SVM 训练...")
        model = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            probability=SVM_PROBABILITY,
            random_state=RANDOM_STATE,
            cache_size=500,
            verbose=False
        )
        model.fit(X_train, y_train)
        joblib.dump(model, SVM_MODEL_PATH)
        print(f"✅ SVM 模型已保存至 {SVM_MODEL_PATH}")

        y_pred = model.predict(X_test)
        model_type = "SVM"

    print(f"✅ 训练完成，耗时：{time.time() - train_start:.2f} 秒")

    # -------------------- 7. 模型评估 --------------------
    print("\n[步骤 7/7] 模型效果评估...")
    # 计算指标
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "=" * 50)
    print(f"  📊 最终评估结果（{model_type}）")
    print("=" * 50)
    print(f"测试集准确率：{test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"宏平均F1值：  {macro_f1:.4f}")
    print(f"加权F1值：    {weighted_f1:.4f}")

    # 详细报告
    print("\n📋 详细分类报告：")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("🧮 混淆矩阵：")
    print(cm)

    # 保存向量化器
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    print(f"\n✅ 向量化器已保存至 {VECTORIZER_SAVE_PATH}")

    # 绘制混淆矩阵图
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.title(f'情感分类混淆矩阵 ({model_type})', fontsize=14)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        cm_img_path = os.path.join(MODEL_DIR, f'confusion_matrix_{model_type.lower()}.png')
        plt.savefig(cm_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 混淆矩阵图已保存：{cm_img_path}")
    except Exception as e:
        print(f"⚠️  绘图失败：{str(e)}")

    # 收尾
    total_time = time.time() - total_start
    print("\n" + "=" * 65)
    print(f"🎉 全部流程执行完成！总耗时：{total_time:.2f} 秒")
    print("=" * 65)

if __name__ == '__main__':
    main()