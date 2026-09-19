# -*- coding: utf-8 -*-
"""
Flask 主程序 - 情感分析系统（规则优先 + 轻量级 XGBoost 增强版）
整合：规则硬拦截（最高优先级） + XGBoost（主模型） + SVM（备用）
功能：单句/批量分析、智能客服、语义检索、反馈闭环、热更新
模型大小 < 50MB，适合轻量部署
"""

# ========== 【必须放在最前面】日志与警告初始化 ==========
import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
jieba_logger = logging.getLogger('jieba')
jieba_logger.setLevel(logging.WARNING)
jieba_logger.propagate = False
jieba_logger.handlers.clear()
logging.basicConfig(level=logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'train'))

# ========== 第三方库 ==========
import csv
import time
import random
import traceback
import threading
from datetime import datetime
import numpy as np

import pandas as pd
import jieba
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import joblib

# 尝试导入 XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ========== 项目模块 ==========
from config import *
from train.preprocessing import preprocess_single
try:
    from train.feature_engineering import transform_extra_features
    FEATURE_ENG_AVAILABLE = True
except ImportError:
    FEATURE_ENG_AVAILABLE = False

# ========== Flask 应用 ==========
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

try:
    from flask_cors import CORS
    if ENABLE_CORS:
        CORS(app)
        print("[信息] CORS 已启用")
except ImportError:
    pass

# ========== 全局变量 ==========
svm_model = None
tfidf_vectorizer = None
xgboost_model = None
label_encoder = None
qa_knowledge_base = None
embedding_model = None
qa_embeddings = None
qa_questions_list = []
qa_answers_list = []
RETRAIN_LOCK = threading.Lock()

# ========== 规则体系（硬拦截） ==========
def apply_rules(text, processed_text):
    """
    应用硬编码规则（正向/中性/负面白名单 + 关键词匹配）
    若命中规则则返回结果字典，否则返回 None
    """
    text_lower = text.lower()

    # ---------- 正向白名单 ----------
    POSITIVE_PHRASES = {
        '很开心', '风景美', '风景漂亮', '非常满意', '强烈推荐',
        '太好了', '很棒', '惊艳', '完美', '超值', '很满意', '好评',
        '没让我失望', '态度好', '服务周到'
    }
    for phrase in POSITIVE_PHRASES:
        if phrase in text_lower:
            return {
                'label': 'positive',
                'label_cn': '正面😊',
                'confidence': 0.92,
                'probabilities': {'positive': 0.92, 'negative': 0.03, 'neutral': 0.05}
            }

    # ---------- 中性白名单 ----------
    NEUTRAL_PHRASES = {
        '包装完好', '没拆开', '后续追评', '帮别人买', '帮朋友买', '代买的',
        '还没用', '先评价', '目前没问题', '正常使用', '普通产品',
        '和描述差不多', '中规中矩', '一般般', '无功无过', '没特别感觉',
        '没问题', '普通的', '没特别', '给朋友买的', '常规购买', '符合预期'
    }
    for phrase in NEUTRAL_PHRASES:
        if phrase in text_lower:
            return {
                'label': 'neutral',
                'label_cn': '中性😐',
                'confidence': 0.85,
                'probabilities': {'positive': 0.05, 'negative': 0.10, 'neutral': 0.85}
            }

    # ---------- 负面词库 ----------
    NEGATIVE_PHRASES = {
        '太差', '太烂', '垃圾', '差劲', '糟糕', '恶心', '废物',
        '没用', '辣鸡', '坑人', '不好', '差评', '失望', '无语', '坑爹',
        '太慢', '等很久', '等一上午', '排队太久', '卡顿', '经常闪退', '很卡',
        '不值', '不值价', '太贵', '效果差', '没效果', '浪费钱',
        '强制购物', '态度差', '不耐烦', '卫生差', '灰尘多', '翻译烂', '错字多',
        '印刷差', '续航差', '拍照模糊', '人太多', '全是人', '尺寸不符', '没法用'
    }
    for phrase in NEGATIVE_PHRASES:
        if phrase in text_lower:
            return {
                'label': 'negative',
                'label_cn': '负面😠',
                'confidence': 0.92,
                'probabilities': {'positive': 0.03, 'negative': 0.92, 'neutral': 0.05}
            }

    # ---------- 关键词匹配（兜底） ----------
    STRONG_NEGATIVE_WORDS = {
        '差', '烂', '垃圾', '差劲', '糟糕', '恶心', '废物',
        '没用', '辣鸡', '坑', '卡', '慢', '贵', '脏', '挤'
    }
    word_set = set(processed_text.split())
    if word_set & STRONG_NEGATIVE_WORDS:
        return {
            'label': 'negative',
            'label_cn': '负面😠',
            'confidence': 0.88,
            'probabilities': {'positive': 0.07, 'negative': 0.88, 'neutral': 0.05}
        }

    return None  # 未命中任何规则

# ==================== 模型预测函数 ====================
def _predict_with_xgboost(text, processed_text):
    """使用 XGBoost 预测（依赖全局 label_encoder）"""
    global label_encoder
    # 构建特征
    tfidf_vec = tfidf_vectorizer.transform([processed_text])
    if FEATURE_ENG_AVAILABLE and ENABLE_EXTRA_FEATURES:
        extra_df = transform_extra_features([text])
        X = hstack([tfidf_vec, extra_df.values])
    else:
        X = tfidf_vec

    dtest = xgb.DMatrix(X)
    proba = xgboost_model.predict(dtest)  # shape (1, n_classes)
    if label_encoder is not None:
        classes = label_encoder.classes_
        idx = np.argmax(proba, axis=1)[0]
        pred_label = classes[idx]
        confidence = float(proba[0][idx])
        prob_dict = {cls: float(proba[0][i]) for i, cls in enumerate(classes)}
    else:
        # 降级方案：假设类别顺序与训练时一致（必须存在 label_encoder）
        raise ValueError("label_encoder 未加载")

    # 置信度红线
    if confidence < 0.55:
        pred_label = 'neutral'
        prob_dict = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.95}
    return {
        'label': pred_label,
        'label_cn': SENTIMENT_MAP.get(pred_label, pred_label),
        'confidence': round(confidence, 4),
        'probabilities': prob_dict
    }

def _predict_with_svm(text, processed_text):
    """SVM 预测（无额外规则，仅模型）"""
    if not svm_model or not tfidf_vectorizer:
        return {
            'label': 'neutral',
            'label_cn': '中性😐',
            'confidence': 0.0,
            'probabilities': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        }
    # 构建特征
    tfidf_vec = tfidf_vectorizer.transform([processed_text])
    if FEATURE_ENG_AVAILABLE and ENABLE_EXTRA_FEATURES:
        extra_df = transform_extra_features([text])
        X = hstack([tfidf_vec, extra_df.values])
    else:
        X = tfidf_vec

    label = svm_model.predict(X)[0]
    proba = svm_model.predict_proba(X)[0]
    prob_dict = {}
    for i, cls in enumerate(svm_model.classes_):
        prob_dict[cls] = float(proba[i])
    confidence = max(proba)
    if confidence < 0.55:
        label = 'neutral'
        prob_dict = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.95}
    label_cn = SENTIMENT_MAP.get(label, label)
    return {
        'label': label,
        'label_cn': label_cn,
        'confidence': round(confidence, 4),
        'probabilities': prob_dict
    }

def predict_sentiment(text):
    """
    统一预测入口：规则优先 → XGBoost → SVM → 默认中性
    """
    if not text or not text.strip():
        return {
            'label': 'neutral',
            'label_cn': '中性😐',
            'confidence': 0.0,
            'probabilities': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        }

    processed = preprocess_single(text)
    if not processed.strip():
        return {
            'label': 'neutral',
            'label_cn': '中性😐',
            'confidence': 0.0,
            'probabilities': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        }

    # 1. 硬规则拦截（最高优先级）
    rule_result = apply_rules(text, processed)
    if rule_result is not None:
        return rule_result

    # 2. XGBoost
    if xgboost_model is not None and tfidf_vectorizer is not None:
        try:
            return _predict_with_xgboost(text, processed)
        except Exception as e:
            print(f"[警告] XGBoost 预测失败，降级到 SVM: {e}")

    # 3. SVM
    if svm_model is not None and tfidf_vectorizer is not None:
        return _predict_with_svm(text, processed)

    # 4. 无模型
    return {
        'label': 'neutral',
        'label_cn': '中性😐',
        'confidence': 0.0,
        'probabilities': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    }

# ========== 关键词提取、QA 等辅助函数 ==========
def extract_keywords(text, top_n=5):
    if not text or not tfidf_vectorizer:
        return []
    processed = preprocess_single(text)
    words = processed.split()
    seen = set()
    result = []
    for w in words:
        if w not in seen and w not in STOP_WORDS:
            seen.add(w)
            result.append(w)
        if len(result) >= top_n:
            break
    return result

def extract_batch_keywords(texts, top_n=10):
    if not texts or not tfidf_vectorizer:
        return []
    word_freq = {}
    for text in texts:
        processed = preprocess_single(text)
        if not processed:
            continue
        for word in processed.split():
            if len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [{'word': w, 'count': c} for w, c in sorted_words[:top_n]]

def is_knowledge_question(text):
    for kw in KNOWLEDGE_KEYWORDS:
        if kw in text:
            return True
    return False

def save_unknown_question(question, similarity):
    try:
        os.makedirs(os.path.dirname(QA_UNKNOWN_QUESTIONS_PATH), exist_ok=True)
        exists = os.path.exists(QA_UNKNOWN_QUESTIONS_PATH)
        with open(QA_UNKNOWN_QUESTIONS_PATH, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['timestamp', 'question', 'similarity'])
            w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), question, round(similarity, 4)])
    except Exception as e:
        print(f"[警告] 保存未知问题失败: {e}")

# ==================== 反馈闭环 ====================
def save_feedback(text, predicted_label, corrected_label, confidence):
    try:
        os.makedirs(os.path.dirname(FEEDBACK_CSV_PATH), exist_ok=True)
        file_exists = os.path.exists(FEEDBACK_CSV_PATH)
        with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'text', 'predicted', 'corrected', 'confidence'])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                text,
                predicted_label,
                corrected_label,
                round(confidence, 4)
            ])
        return True
    except Exception as e:
        print(f"[反馈错误] {e}")
        return False

def get_feedback_data():
    if not os.path.exists(FEEDBACK_CSV_PATH):
        return pd.DataFrame(columns=['text', 'corrected'])
    df = pd.read_csv(FEEDBACK_CSV_PATH, encoding='utf-8-sig')
    if 'text' in df.columns and 'corrected' in df.columns:
        return df[['text', 'corrected']].dropna()
    return pd.DataFrame(columns=['text', 'corrected'])

def trigger_retrain():
    def _task():
        with RETRAIN_LOCK:
            print("[重训] 开始后台模型更新...")
            try:
                retrain_model()
            except Exception as e:
                print(f"[重训错误] {e}")
                traceback.print_exc()
    thread = threading.Thread(target=_task)
    thread.daemon = True
    thread.start()

def retrain_model():
    global xgboost_model, svm_model, tfidf_vectorizer, label_encoder
    # 加载原始数据
    train_texts = []
    train_labels = []
    if os.path.exists(ORIGINAL_TRAIN_DATA_PATH):
        original_df = pd.read_csv(ORIGINAL_TRAIN_DATA_PATH, encoding='utf-8-sig')
        if 'text' in original_df.columns and 'label' in original_df.columns:
            train_texts.extend(original_df['text'].tolist())
            train_labels.extend(original_df['label'].tolist())
            print(f"[重训] 加载原始数据 {len(original_df)} 条")

    feedback_df = get_feedback_data()
    if len(feedback_df) == 0:
        print("[重训] 无反馈数据，跳过")
        return False

    feedback_texts = feedback_df['text'].tolist()
    feedback_labels = feedback_df['corrected'].tolist()
    print(f"[重训] 加载反馈数据 {len(feedback_df)} 条")

    all_texts = train_texts + feedback_texts
    all_labels = train_labels + feedback_labels
    if len(all_texts) < 10:
        print("[重训] 数据量不足")
        return False

    processed_texts = [preprocess_single(t) for t in all_texts]
    valid_pairs = [(p, l) for p, l in zip(processed_texts, all_labels) if p.strip()]
    if not valid_pairs:
        return False
    final_texts, final_labels = zip(*valid_pairs)

    # TF-IDF
    new_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_tfidf = new_vectorizer.fit_transform(final_texts)

    # 额外特征
    if FEATURE_ENG_AVAILABLE and ENABLE_EXTRA_FEATURES:
        extra_df = transform_extra_features(final_texts)
        X = hstack([X_tfidf, extra_df.values])
    else:
        X = X_tfidf

    # 训练 XGBoost（主）
    if XGB_AVAILABLE:
        le = LabelEncoder()
        y_enc = le.fit_transform(final_labels)
        dtrain = xgb.DMatrix(X, label=y_enc)
        params = XGBOOST_PARAMS.copy()
        params['num_class'] = len(le.classes_)
        params['objective'] = 'multi:softprob'
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 200),
            verbose_eval=False
        )
        joblib.dump(model, XGBOOST_MODEL_PATH)
        joblib.dump(le, LABEL_ENCODER_PATH)
        print(f"[重训] XGBoost 模型已保存至 {XGBOOST_MODEL_PATH}")
        xgboost_model = model
        label_encoder = le
    else:
        # 降级 SVM
        model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        model.fit(X, final_labels)
        joblib.dump(model, SVM_MODEL_PATH)
        print(f"[重训] SVM 模型已保存至 {SVM_MODEL_PATH}")
        svm_model = model

    # 更新向量化器
    tfidf_vectorizer = new_vectorizer
    joblib.dump(new_vectorizer, TFIDF_VECTORIZER_PATH)

    # 清空反馈
    if os.path.exists(FEEDBACK_CSV_PATH):
        os.remove(FEEDBACK_CSV_PATH)
    print("[重训] 反馈日志已清空，热更新完成！")
    return True

# ==================== 加载模型 ====================
def load_model():
    global svm_model, tfidf_vectorizer, xgboost_model, qa_knowledge_base, label_encoder

    print("[信息] 正在加载模型资源...")

    # 1. 加载 XGBoost
    if os.path.exists(XGBOOST_MODEL_PATH) and XGB_AVAILABLE:
        try:
            xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
            le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
            if os.path.exists(le_path):
                label_encoder = joblib.load(le_path)
            print("[信息] XGBoost 主模型加载成功")
        except Exception as e:
            xgboost_model = None
            print(f"[警告] XGBoost 加载失败: {e}")

    # 2. 加载 SVM（备用）
    model_path = SVM_MODEL_PATH
    if not os.path.exists(model_path):
        alt_path = os.path.join(MODEL_DIR, 'svm_sentiment_model.pkl')
        if os.path.exists(alt_path):
            model_path = alt_path
    if os.path.exists(model_path):
        try:
            svm_model = joblib.load(model_path)
            print(f"[信息] SVM 备用模型加载成功")
        except Exception as e:
            svm_model = None
            print(f"[警告] SVM 加载失败: {e}")

    # 3. 加载 TF-IDF
    vec_path = TFIDF_VECTORIZER_PATH
    if not os.path.exists(vec_path):
        alt_vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
        if os.path.exists(alt_vec_path):
            vec_path = alt_vec_path
    if os.path.exists(vec_path):
        try:
            tfidf_vectorizer = joblib.load(vec_path)
            print(f"[信息] TF-IDF 向量化器加载成功")
        except Exception as e:
            tfidf_vectorizer = None
            print(f"[警告] 向量化器加载失败: {e}")

    # 4. 加载知识库
    if os.path.exists(QA_KNOWLEDGE_BASE_PATH):
        qa_knowledge_base = pd.read_csv(QA_KNOWLEDGE_BASE_PATH)
        print(f"[信息] 问答知识库加载成功，共 {len(qa_knowledge_base)} 条")
        load_embedding_model()
    else:
        print(f"[提示] 问答知识库不存在")

    if xgboost_model is None and svm_model is None:
        print("[错误] 没有任何模型可用！请先训练模型。")
    else:
        print("[信息] 资源加载完成")
    return True

def load_embedding_model():
    global embedding_model, qa_embeddings, qa_questions_list, qa_answers_list
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[警告] sentence-transformers 未安装，语义检索禁用")
        return False
    try:
        local_path = os.path.join(MODEL_DIR, 'embedding_model')
        embedding_model = SentenceTransformer(local_path)
        qa_questions_list = qa_knowledge_base['question'].tolist()
        qa_answers_list = qa_knowledge_base['answer'].tolist()
        qa_embeddings = embedding_model.encode(qa_questions_list, normalize_embeddings=True)
        print("[信息] 语义检索模型加载成功")
        return True
    except Exception as e:
        print(f"[警告] 语义检索加载失败: {e}")
        return False

def find_best_answer(question):
    if qa_knowledge_base is None or len(qa_knowledge_base) == 0:
        return None, 0.0
    if embedding_model is not None:
        q_emb = embedding_model.encode([question], normalize_embeddings=True)
        sim = cosine_similarity(q_emb, qa_embeddings)[0]
        idx = sim.argmax()
        return qa_answers_list[idx], float(sim[idx])
    # 降级 Jaccard
    words = set(jieba.lcut(question))
    best_sim = 0.0
    best_ans = None
    for _, row in qa_knowledge_base.iterrows():
        qw = set(jieba.lcut(str(row['question'])))
        union = len(words | qw)
        sim = 0.0 if union == 0 else len(words & qw) / union
        if sim > best_sim:
            best_sim = sim
            best_ans = row['answer']
    return best_ans, best_sim

# ==================== Flask 路由 ====================
@app.route('/')
def index():
    return render_template('index.html', project_name=PROJECT_NAME, project_version=PROJECT_VERSION)

@app.route('/api/single_sentiment', methods=['POST'])
def single_sentiment():
    try:
        data = request.get_json() or request.form
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'success': False, 'message': '请输入文本'}), 400
        result = predict_sentiment(text)
        keywords = extract_keywords(text, top_n=5)
        return jsonify({
            'success': True,
            'data': {'text': text, 'sentiment': result, 'keywords': keywords}
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/batch_sentiment', methods=['POST'])
def batch_sentiment():
    try:
        df = None
        text_column = 'text'
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'message': '请选择文件'}), 400
            import io
            file.seek(0)
            raw_bytes = file.read()
            decoded_text = None
            for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']:
                try:
                    decoded_text = raw_bytes.decode(enc)
                    break
                except:
                    continue
            if decoded_text is None:
                decoded_text = raw_bytes.decode('gbk', errors='replace')
            for sep in [',', '\t', ';']:
                try:
                    df = pd.read_csv(io.StringIO(decoded_text), sep=sep, on_bad_lines='skip')
                    if len(df.columns) >= 1:
                        break
                except:
                    continue
            if df is None or len(df.columns) == 0:
                lines = decoded_text.splitlines()
                df = pd.DataFrame({'text': lines})
            if text_column not in df.columns:
                possible_cols = ['text', '内容', '评论', 'content', '文本', 'review']
                for col in possible_cols:
                    if col in df.columns:
                        text_column = col
                        break
                else:
                    return jsonify({'success': False, 'message': '未找到文本列'}), 400
        else:
            data = request.get_json() or request.form
            texts = data.get('texts', [])
            if not texts:
                return jsonify({'success': False, 'message': '请提供文本列表'}), 400
            df = pd.DataFrame({text_column: texts})

        results = []
        for _, row in df.iterrows():
            result = predict_sentiment(str(row[text_column]))
            results.append(result)
        df['sentiment'] = [r['label'] for r in results]
        df['sentiment_cn'] = [r['label_cn'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]

        total = len(df)
        counts = df['sentiment'].value_counts().to_dict()
        for label in SENTIMENT_LABELS:
            counts.setdefault(label, 0)
        ratios = {l: round(counts.get(l,0)/total, 4) for l in SENTIMENT_LABELS}
        pos_texts = df[df['sentiment']=='positive'][text_column].tolist()
        neg_texts = df[df['sentiment']=='negative'][text_column].tolist()
        pos_kw = extract_batch_keywords(pos_texts, 10)
        neg_kw = extract_batch_keywords(neg_texts, 10)
        preview = df.head(50).to_dict('records')
        return jsonify({
            'success': True,
            'data': {
                'total': total,
                'sentiment_counts': counts,
                'sentiment_ratios': ratios,
                'positive_keywords': pos_kw,
                'negative_keywords': neg_kw,
                'preview': preview,
                'columns': df.columns.tolist()
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/export_result', methods=['POST'])
def export_result():
    try:
        data = request.get_json() or request.form
        results = data.get('results', [])
        if not results:
            return jsonify({'success': False, 'message': '无数据'}), 400
        df = pd.DataFrame(results)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f'sentiment_result_{int(time.time())}.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        return send_file(path, as_attachment=True, download_name='情感分析结果.csv', mimetype='text/csv')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/qa_chat', methods=['POST'])
def qa_chat():
    try:
        data = request.get_json() or request.form
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'success': False, 'message': '请输入问题'}), 400
        sentiment_result = predict_sentiment(question)
        is_negative = (sentiment_result['label'] == 'negative' and
                       sentiment_result['confidence'] >= NEGATIVE_CONFIDENCE_THRESHOLD)
        is_knowledge = is_knowledge_question(question)
        answer, sim = find_best_answer(question)
        parts = []
        comforted = False
        if answer and sim >= QA_SIMILARITY_THRESHOLD:
            if is_negative and not is_knowledge:
                parts.append(random.choice(NEGATIVE_COMFORT_MESSAGES))
                comforted = True
            parts.append(answer)
        else:
            if is_negative:
                parts.append("非常抱歉给您带来不好的体验，您的反馈我们已经认真记录，后续会持续优化改进，感谢您的宝贵意见~")
                comforted = True
            else:
                parts.append("抱歉，我暂时无法回答您的问题。我会把您的问题记录下来，后续会不断优化的~")
            save_unknown_question(question, sim)
        final_answer = '\n\n'.join(parts)
        return jsonify({
            'success': True,
            'data': {
                'question': question,
                'answer': final_answer,
                'similarity': round(sim, 4),
                'is_hit': sim >= QA_SIMILARITY_THRESHOLD,
                'sentiment': sentiment_result,
                'is_comforted': comforted
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json() or request.form
        text = data.get('text', '').strip()
        predicted = data.get('predicted_label', '')
        corrected = data.get('corrected_label', '')
        confidence = float(data.get('confidence', 0.0))
        if not text or not corrected:
            return jsonify({'success': False, 'message': '缺少参数'}), 400
        save_feedback(text, predicted, corrected, confidence)
        count = len(get_feedback_data())
        triggered = count >= RETRAIN_THRESHOLD
        if triggered:
            trigger_retrain()
        return jsonify({
            'success': True,
            'message': '感谢您的反馈！',
            'data': {'feedback_count': count, 'retrain_triggered': triggered}
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/retrain/manual', methods=['POST'])
def manual_retrain():
    try:
        trigger_retrain()
        return jsonify({'success': True, 'message': '已触发后台重训'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'project': PROJECT_NAME,
        'version': PROJECT_VERSION,
        'xgboost_loaded': xgboost_model is not None,
        'svm_loaded': svm_model is not None,
        'embedding_loaded': embedding_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'message': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'message': '服务器内部错误'}), 500

# ==================== 启动 ====================
if __name__ == '__main__':
    init_dirs()
    print("=" * 60)
    print(f"  {PROJECT_NAME} v{PROJECT_VERSION}")
    print("=" * 60)
    load_model()
    print(f"\n访问地址: http://localhost:{FLASK_PORT}")
    print(f"  主模型: {'XGBoost' if xgboost_model else 'SVM（备用）'}")
    print(f"  语义检索: {'已启用' if embedding_model else '未启用'}")
    print(f"  特征工程: {'已启用' if FEATURE_ENG_AVAILABLE and ENABLE_EXTRA_FEATURES else '未启用'}")
    print("=" * 60)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)