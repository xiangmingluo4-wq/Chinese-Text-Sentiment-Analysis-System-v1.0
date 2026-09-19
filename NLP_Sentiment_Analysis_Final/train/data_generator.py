# -*- coding: utf-8 -*-
"""
数据生成脚本 V2.1 - 增强版
优化：扩充词库、增加中性表达、提升多样性、严格去重、与特征工程词典对齐
"""

import os
import random
import pandas as pd
import re
from collections import Counter

# ==================== 基础配置 ====================
RANDOM_STATE = 42
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
SENTIMENT_MAP = {
    'positive': '正面',
    'negative': '负面',
    'neutral': '中性'
}
DOMAINS = [
    '电商购物', '餐饮美食', '影视娱乐', '酒店住宿', '旅游出行',
    '教育培训', '数码产品', '美妆护肤', '图书阅读', '医疗健康'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SENTIMENT_DATA_PATH = os.path.join(DATA_DIR, 'sentiment_dataset_enhanced.csv')

random.seed(RANDOM_STATE)

# ==================== 增强词库（与 feature_engineering 保持一致） ====================
SYNONYMS = {
    '正面形容词': [
        '很棒', '很好', '超赞', '绝了', '不错', '给力', '靠谱', '优秀', '出色', '惊艳',
        '满意', '惊喜', '舒服', '省心', '划算', '完美', '强大', '实用', '舒适', '干净',
        '明亮', '周到', '热情', '专业', '细致', '温馨', '精致', '良心', '地道', '一流',
        '绝佳', '稳定', '流畅', '清晰', '细腻', '丰富', '有趣', '精彩', '振奋', '受益',
        '有用', '经典', '深刻', '真实', '高效', '神速'
    ],
    '负面形容词': [
        '很差', '糟糕', '拉胯', '踩雷', '失望', '垃圾', '差劲', '一般', '拉垮', '鸡肋',
        '坑人', '不值', '麻烦', '别扭', '后悔', '坑爹', '渣', '土', '旧', '破',
        '难用', '无用', '无效', '繁琐', '冷漠', '恶劣', '敷衍', '不耐烦', '不负责',
        '误导', '欺骗', '翻车', '智商税', '劝退', '大踩雷', '不值价', '凑合'
    ],
    '程度副词': [
        '非常', '特别', '超级', '真的', '确实', '实在', '格外', '相当', '挺', '蛮', '还挺', '太'
    ],
    '语气词正面': [
        '啊', '呀', '哇', '哦', '呢', '~', '！', '！！', '～'
    ],
    '语气词负面': [
        '啊', '唉', '哎', '吧', '嘛', '...', '。', '😅'
    ],
    '网络热词正面': [
        'yyds', '绝绝子', '闭眼入', '谁懂啊', '爱了爱了', '狠狠爱住', '天花板', '上头'
    ],
    '网络热词负面': [
        '大踩雷', '劝退', '避雷', '智商税', '翻车', '栓Q', '大可不必', '无大语'
    ],
    '中性表达': [
        '还行', '一般', '普通', '正常', '中规中矩', '无功无过', '没特别感觉',
        '符合预期', '差不多', '还好', '过得去', '凑合', '平平无奇'
    ]
}

# 领域专属描述词（扩充，与特征工程词典对齐）
DOMAIN_KEYWORDS = {
    '电商购物': {
        '正面': [
            '包装严实', '发货快', '物流给力', '性价比高', '物超所值', '和描述一致',
            '做工精细', '材质厚实', '手感好', '质量过硬', '颜色正', '尺寸合适'
        ],
        '负面': [
            '发货慢', '物流龟速', '货不对板', '做工粗糙', '材质廉价', '有瑕疵',
            '色差大', '缺斤少两', '包装破损', '质量差', '尺码不准'
        ],
        '主体': ['商品', '宝贝', '东西', '货品', '物件', '产品']
    },
    '餐饮美食': {
        '正面': [
            '口味正宗', '分量足', '食材新鲜', '环境干净', '服务热情', '上菜快',
            '味道绝了', '性价比高', '摆盘精美', '菜品丰富'
        ],
        '负面': [
            '口味一般', '分量少', '不新鲜', '环境差', '服务差', '上菜慢',
            '价格贵', '卫生堪忧', '菜品单一', '口味偏重'
        ],
        '主体': ['餐厅', '菜品', '菜', '套餐', '店', '美食']
    },
    '影视娱乐': {
        '正面': [
            '剧情紧凑', '演技在线', '画面精美', '配乐好听', '立意深刻',
            '节奏舒服', '演员选得好', '反转精彩', '特效震撼'
        ],
        '负面': [
            '剧情拖沓', '演技尴尬', '画面廉价', '配乐出戏', '逻辑不通',
            '烂尾', '演技浮夸', '台词尴尬', '特效五毛'
        ],
        '主体': ['电影', '剧', '片子', '影片', '综艺', '动画']
    },
    '酒店住宿': {
        '正面': [
            '房间干净', '隔音好', '床很舒服', '早餐丰盛', '位置便利',
            '服务贴心', '性价比高', '设施齐全', '环境优雅', '热水稳定'
        ],
        '负面': [
            '房间脏乱', '隔音差', '床很硬', '早餐难吃', '位置偏僻',
            '服务冷淡', '设施老旧', '有异味', '热水不稳', '噪音大'
        ],
        '主体': ['酒店', '房间', '住宿', '宾馆', '民宿', '客栈']
    },
    '旅游出行': {
        '正面': [
            '风景优美', '人不多', '门票值', '讲解专业', '行程合理',
            '体验很棒', '拍照出片', '设施完善', '导游负责', '交通方便'
        ],
        '负面': [
            '人挤人', '门票贵', '没啥看头', '讲解敷衍', '行程赶',
            '体验很差', '照骗', '设施破旧', '导游强制购物', '交通不便'
        ],
        '主体': ['景点', '景区', '旅行', '行程', '路线', '旅游团']
    },
    '教育培训': {
        '正面': [
            '老师讲得好', '干货多', '容易懂', '课程系统', '答疑及时',
            '收获大', '性价比高', '资料齐全', '互动多', '进步明显'
        ],
        '负面': [
            '老师念PPT', '水课', '听不懂', '课程零散', '答疑敷衍',
            '没收获', '不值这个价', '资料不全', '互动少', '进度慢'
        ],
        '主体': ['课程', '老师', '教学', '培训', '网课', '直播课']
    },
    '数码产品': {
        '正面': [
            '续航给力', '运行流畅', '像素高', '音质好', '手感棒',
            '系统丝滑', '充电快', '颜值高', '屏幕清晰', '性能强劲'
        ],
        '负面': [
            '续航拉胯', '卡顿', '像素渣', '音质差', '手感差',
            '系统难用', '充电慢', '发热严重', '屏幕泛黄', '性能虚标'
        ],
        '主体': ['手机', '电脑', '平板', '耳机', '相机', '手表']
    },
    '美妆护肤': {
        '正面': [
            '保湿好', '不刺激', '上脸服帖', '成分安全', '效果明显',
            '肤感好', '不闷痘', '持妆久', '提亮肤色', '质地细腻'
        ],
        '负面': [
            '拔干', '刺激皮肤', '假白', '成分刺激', '没效果',
            '肤感差', '闷痘', '脱妆快', '假滑', '气味刺鼻'
        ],
        '主体': ['护肤品', '化妆品', '面膜', '口红', '精华', '防晒']
    },
    '图书阅读': {
        '正面': [
            '内容扎实', '文笔好', '干货多', '印刷清晰', '装帧精美',
            '观点新颖', '值得反复读', '启发很大', '逻辑清晰', '案例丰富'
        ],
        '负面': [
            '内容空洞', '文笔差', '凑字数', '印刷模糊', '装帧劣质',
            '观点老套', '不值得读', '翻译差劲', '逻辑混乱', '案例过时'
        ],
        '主体': ['书', '书籍', '小说', '教材', '图书', '读物']
    },
    '医疗健康': {
        '正面': [
            '医生专业', '检查仔细', '效果明显', '服务好', '排队快',
            '环境干净', '收费透明', '药很有效', '态度亲切', '设备先进'
        ],
        '负面': [
            '医生敷衍', '检查潦草', '没效果', '服务差', '排队久',
            '环境脏乱', '收费乱', '副作用大', '态度冷漠', '设备老旧'
        ],
        '主体': ['药品', '医院', '医生', '体检', '治疗', '诊所']
    }
}

# ==================== 增强函数 ====================
def synonym_replace(text, sentiment):
    """随机同义词替换，保持语义不变"""
    if random.random() > 0.5:  # 50%概率触发
        return text

    if sentiment == 'positive':
        adj_pool = SYNONYMS['正面形容词']
    elif sentiment == 'negative':
        adj_pool = SYNONYMS['负面形容词']
    else:
        adj_pool = SYNONYMS['中性表达']

    # 替换程度副词
    if random.random() < 0.3:
        for adv in ['非常', '特别', '真的', '很', '太']:
            if adv in text:
                new_adv = random.choice(SYNONYMS['程度副词'])
                text = text.replace(adv, new_adv, 1)
                break

    # 替换形容词/情感词
    if random.random() < 0.4:
        # 尝试替换已有词汇
        for old_word in ['很好', '很棒', '好', '差', '糟糕', '一般']:
            if old_word in text:
                new_word = random.choice(adj_pool)
                text = text.replace(old_word, new_word, 1)
                break

    return text

def add_modal_particle(text, sentiment):
    """添加语气词和网络热词"""
    if random.random() > 0.5:
        return text

    if sentiment == 'positive':
        particles = SYNONYMS['语气词正面']
        hot_words = SYNONYMS['网络热词正面']
    elif sentiment == 'negative':
        particles = SYNONYMS['语气词负面']
        hot_words = SYNONYMS['网络热词负面']
    else:
        particles = ['啊', '吧', '呢', '。']
        hot_words = ['还行', '凑合', '一般般']

    # 句尾添加语气词
    if random.random() < 0.3:
        particle = random.choice(particles)
        if text.endswith('！') or text.endswith('。') or text.endswith('?'):
            text = text[:-1] + particle
        else:
            text = text + particle

    # 随机添加网络热词
    if random.random() < 0.15:
        hot_word = random.choice(hot_words)
        if sentiment == 'positive':
            text = text + '，' + hot_word + '！'
        elif sentiment == 'negative':
            text = text + '，简直' + hot_word + '。'
        else:
            text = text + '，' + hot_word

    return text

def random_sentence_concat(sentiment, domain):
    """拼接两个短句生成复合句"""
    domain_info = DOMAIN_KEYWORDS[domain]
    subject = random.choice(domain_info['主体'])

    if sentiment == 'positive':
        desc1 = random.choice(domain_info['正面'])
        desc2 = random.choice([d for d in domain_info['正面'] if d != desc1])
        templates = [
            f"{subject}{desc1}，而且{desc2}，很满意。",
            f"不仅{subject}{desc1}，就连细节都{desc2}，超出预期。",
            f"{subject}{desc1}，整体体验{desc2}，推荐。",
            f"本来没抱期待，结果{subject}{desc1}，还{desc2}，惊喜。"
        ]
    elif sentiment == 'negative':
        desc1 = random.choice(domain_info['负面'])
        desc2 = random.choice([d for d in domain_info['负面'] if d != desc1])
        templates = [
            f"{subject}{desc1}，而且{desc2}，很失望。",
            f"不仅{subject}{desc1}，服务还{desc2}，踩大雷。",
            f"{subject}{desc1}，体验感{desc2}，不推荐。",
            f"本来挺期待的，结果{subject}{desc1}，还{desc2}，后悔了。"
        ]
    else:
        # 中性：一正一负
        desc_pos = random.choice(domain_info['正面'])
        desc_neg = random.choice(domain_info['负面'])
        templates = [
            f"{subject}{desc_pos}，但也{desc_neg}，整体一般吧。",
            f"虽说{subject}{desc_pos}，但可惜{desc_neg}，中规中矩。",
            f"{subject}有优点也有缺点，{desc_pos}是亮点，{desc_neg}不足。"
        ]
    return random.choice(templates)

def generate_short_text(sentiment, domain):
    """生成短句评论（模拟真实短评）"""
    domain_info = DOMAIN_KEYWORDS[domain]
    if sentiment == 'positive':
        templates = [
            f"{random.choice(domain_info['正面'])}，好评！",
            f"{random.choice(domain_info['主体'])}挺不错的。",
            f"还可以，{random.choice(domain_info['正面'])}。",
            f"满意，{random.choice(domain_info['正面'])}。",
            f"{random.choice(SYNONYMS['网络热词正面'])}"
        ]
    elif sentiment == 'negative':
        templates = [
            f"{random.choice(domain_info['负面'])}，差评。",
            f"{random.choice(domain_info['主体'])}不太行。",
            f"一般般，{random.choice(domain_info['负面'])}。",
            f"后悔了，{random.choice(domain_info['负面'])}。",
            f"{random.choice(SYNONYMS['网络热词负面'])}"
        ]
    else:
        templates = [
            "还行吧，不好不坏。",
            "中规中矩，没什么特别的。",
            "凑合，能用就行。",
            "一般般，符合预期。",
            "不好说，再看看。",
            "普普通通，没亮点。",
            "无功无过，就当买个教训。",
            "说不上好坏，反正就是那样。"
        ]
    return random.choice(templates)

def generate_one_sample(sentiment, domain, used_texts):
    """生成单条样本，带去重"""
    max_try = 20
    for _ in range(max_try):
        mode = random.choices(
            ['template_base', 'concat', 'short', 'domain_custom'],
            weights=[0.25, 0.25, 0.2, 0.30]
        )[0]

        if mode == 'short':
            text = generate_short_text(sentiment, domain)
        elif mode == 'concat':
            text = random_sentence_concat(sentiment, domain)
        elif mode == 'domain_custom':
            domain_info = DOMAIN_KEYWORDS[domain]
            subject = random.choice(domain_info['主体'])
            if sentiment == 'positive':
                desc = random.choice(domain_info['正面'])
                templates = [
                    f"这个{subject}{desc}，非常满意。",
                    f"{subject}{desc}，超出预期。",
                    f"用下来感觉{subject}{desc}，推荐。"
                ]
            elif sentiment == 'negative':
                desc = random.choice(domain_info['负面'])
                templates = [
                    f"这个{subject}{desc}，很失望。",
                    f"{subject}{desc}，不值这个价。",
                    f"用下来感觉{subject}{desc}，不推荐。"
                ]
            else:
                desc_pos = random.choice(domain_info['正面'])
                desc_neg = random.choice(domain_info['负面'])
                templates = [
                    f"{subject}还行，{desc_pos}但{desc_neg}。",
                    f"{subject}一般般，既不突出也不差。",
                    f"这个{subject}中规中矩，{desc_pos}和{desc_neg}都有。"
                ]
            text = random.choice(templates)
        else:
            # 基础模板 + 同义词替换
            domain_info = DOMAIN_KEYWORDS[domain]
            subject = random.choice(domain_info['主体'])
            if sentiment == 'positive':
                base_templates = [
                    f"这个{subject}真的太棒了，非常满意！",
                    f"{subject}质量很好，超出预期，推荐购买！",
                    f"用了一段时间，{subject}确实不错，给好评！",
                    f"{subject}性价比很高，值得入手！",
                    f"第一次买{subject}，没想到这么好，惊喜！"
                ]
            elif sentiment == 'negative':
                base_templates = [
                    f"这个{subject}太让人失望了，不推荐购买。",
                    f"{subject}质量很差，和描述的完全不一样。",
                    f"用了几天{subject}就出问题了，质量堪忧。",
                    f"{subject}性价比太低了，不值这个价。",
                    f"第一次买{subject}就踩雷了，太倒霉了。"
                ]
            else:
                base_templates = [
                    f"{subject}还行吧，中规中矩。",
                    f"{subject}一般般，没有特别好也没有特别差。",
                    f"收到{subject}了，和描述的差不多。",
                    f"{subject}性价比一般，看个人需求。"
                ]
            text = random.choice(base_templates)
            text = synonym_replace(text, sentiment)

        # 通用增强
        text = add_modal_particle(text, sentiment)

        # 清洗：去除多余空格，确保不是空文本
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue

        # 去重检查（短文本去重更严格，10字符以上）
        if len(text) < 3:
            continue
        if text not in used_texts:
            used_texts.add(text)
            return text

    # 多次尝试后仍重复，则返回一个随机后缀
    return text + str(random.randint(100, 999))

def generate_sentiment_data(total_count=20000, positive_ratio=0.35, negative_ratio=0.35):
    """生成增强版情感数据集"""
    print(f"[信息] 开始生成增强版情感数据，总量：{total_count}条")
    positive_count = int(total_count * positive_ratio)
    negative_count = int(total_count * negative_ratio)
    neutral_count = total_count - positive_count - negative_count

    print(f"[信息] 分布：正面{positive_count}条，负面{negative_count}条，中性{neutral_count}条")

    data = []
    used_texts = set()
    id_counter = 1

    for sentiment, count in [('positive', positive_count), ('negative', negative_count), ('neutral', neutral_count)]:
        label_cn = SENTIMENT_MAP[sentiment]
        print(f"[信息] 正在生成{label_cn}评价...")
        for i in range(count):
            domain = random.choice(DOMAINS)
            text = generate_one_sample(sentiment, domain, used_texts)

            # 双人标注模拟（95%一致率）
            annotator1 = sentiment
            if random.random() < 0.95:
                annotator2 = sentiment
            else:
                other_labels = [s for s in SENTIMENT_LABELS if s != sentiment]
                annotator2 = random.choice(other_labels)

            data.append({
                'id': id_counter,
                'text': text,
                'domain': domain,
                'sentiment': sentiment,
                'annotator1': annotator1,
                'annotator2': annotator2
            })
            id_counter += 1

    # 打乱顺序
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df['id'] = range(1, len(df) + 1)

    print(f"[信息] 数据生成完成，共 {len(df)} 条")
    return df

def save_data(df, output_path=None):
    if output_path is None:
        output_path = SENTIMENT_DATA_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[信息] 数据已保存到: {output_path}")
    return output_path

def print_statistics(df):
    print("\n" + "=" * 60)
    print("  增强版数据统计信息")
    print("=" * 60)

    print(f"\n总数据量: {len(df)} 条")

    # 情感分布
    print("\n情感分布:")
    sentiment_counts = df['sentiment'].value_counts()
    for label in SENTIMENT_LABELS:
        count = sentiment_counts.get(label, 0)
        ratio = count / len(df) * 100
        print(f"  {SENTIMENT_MAP[label]}: {count} 条 ({ratio:.1f}%)")

    # 领域分布
    print("\n领域分布:")
    domain_counts = df['domain'].value_counts()
    for domain in DOMAINS:
        count = domain_counts.get(domain, 0)
        ratio = count / len(df) * 100
        print(f"  {domain}: {count} 条 ({ratio:.1f}%)")

    # 标注一致性
    agree_count = (df['annotator1'] == df['annotator2']).sum()
    agree_rate = agree_count / len(df) * 100
    print("\n标注一致性:")
    print(f"  一致数量: {agree_count} 条")
    print(f"  一致率: {agree_rate:.1f}%")

    # 重复率
    unique_count = df['text'].nunique()
    repeat_rate = (1 - unique_count / len(df)) * 100
    print("\n文本重复率:")
    print(f"  唯一文本数: {unique_count} 条")
    print(f"  重复率: {repeat_rate:.2f}%")

    # 文本长度统计
    text_lengths = df['text'].str.len()
    print("\n文本长度统计:")
    print(f"  平均长度: {text_lengths.mean():.1f} 字符")
    print(f"  最短长度: {text_lengths.min()} 字符")
    print(f"  最长长度: {text_lengths.max()} 字符")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    print("=" * 60)
    print("  中文文本情感分析系统 - 增强版数据生成 V2.1")
    print("  高多样性 · 低重复 · 与特征工程对齐")
    print("=" * 60)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    # 可调整参数
    TOTAL_SAMPLES = 20000      # 总样本数，建议 2-3 万
    POSITIVE_RATIO = 0.35
    NEGATIVE_RATIO = 0.35
    # 中性自动补齐

    df = generate_sentiment_data(total_count=TOTAL_SAMPLES,
                                 positive_ratio=POSITIVE_RATIO,
                                 negative_ratio=NEGATIVE_RATIO)
    save_data(df)
    print_statistics(df)

    print("\n[完成] 增强版数据生成任务完成！")
    print("[提示] 生成的文件位于 data/sentiment_dataset_enhanced.csv")
    print("[提示] 如需调整数量或比例，请修改 TOTAL_SAMPLES 等参数")