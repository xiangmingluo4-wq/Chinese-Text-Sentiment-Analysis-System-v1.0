# -*- coding: utf-8 -*-
"""
情感联动智能客服脚本
中文文本情感分析系统 - 智能问答 + 情感联动安抚
"""
import warnings
# 屏蔽 jieba 依赖的 pkg_resources 弃用警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import os
import sys
import random
import pandas as pd
import jieba
import importlib.util

# ========== 强制加载项目内的 config.py（彻底解决同名冲突） ==========
# 计算项目根目录（code 文件夹的上一级）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 项目配置文件的绝对路径
config_file_path = os.path.join(project_root, "config.py")

# 按指定路径强制加载模块
spec = importlib.util.spec_from_file_location("project_config", config_file_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# 将所有配置变量注入当前作用域（等价于 from config import *）
globals().update({
    k: v for k, v in config_module.__dict__.items()
    if not k.startswith('_')
})

# 验证
print(f"[调试] 已加载配置文件: {config_file_path}")
print(f"[调试] 知识库路径: {QA_KNOWLEDGE_BASE_PATH}")
print("[调试] ✅ 配置变量全部导入成功")

class QAChatbot:
    """
    情感联动智能客服
    """
    
    def __init__(self, knowledge_base_path=None):
        """
        初始化智能客服
        
        Args:
            knowledge_base_path: 知识库文件路径
        """
        self.knowledge_base = None
        self.unknown_questions = []
        
        # 加载知识库
        if knowledge_base_path is None:
            knowledge_base_path = QA_KNOWLEDGE_BASE_PATH
        
        self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, path):
        """
        加载问答知识库
        """
        if os.path.exists(path):
            self.knowledge_base = pd.read_csv(path)
            print(f"[信息] 知识库加载成功，共 {len(self.knowledge_base)} 条问答对")
        else:
            print(f"[提示] 知识库文件不存在: {path}")
            print("[提示] 将使用默认问答对")
            self._init_default_knowledge()
    
    def _init_default_knowledge(self):
        """
        初始化默认知识库
        """
        default_qa = [
            {'question': '你好', 'answer': '你好！很高兴为您服务，请问有什么可以帮助您的吗？'},
            {'question': '您好', 'answer': '您好！请问有什么我可以帮您的？'},
            {'question': '谢谢', 'answer': '不客气！能帮到您是我的荣幸~'},
            {'question': '感谢', 'answer': '感谢您的认可，我们会继续努力的！'},
            {'question': '再见', 'answer': '再见！祝您生活愉快，有问题随时来找我~'},
            {'question': '拜拜', 'answer': '拜拜~ 期待下次再为您服务！'},
            {'question': '这个系统怎么用', 'answer': '本系统支持三个功能：\n1. 批量分析：上传CSV文件，批量分析情感倾向\n2. 单句分析：输入文本，实时分析情感\n3. 智能客服：就是我啦，可以回答您的问题~'},
            {'question': '怎么批量分析', 'answer': '切换到"批量分析"标签页，点击上传区域选择CSV文件，系统会自动分析并展示统计结果和图表。'},
            {'question': '支持什么格式', 'answer': '目前支持CSV格式的文件，文件中需要包含text列作为评论文本。'},
            {'question': '准确率怎么样', 'answer': '我们的模型在测试集上准确率达到95%以上，可以满足大多数场景的需求。'},
            {'question': '模型是怎么训练的', 'answer': '我们使用SVM算法和TF-IDF特征，在10000条标注数据上训练而成，覆盖10个领域。'},
            {'question': '数据是哪里来的', 'answer': '数据是通过模板化生成的，覆盖电商、餐饮、影视等10个领域，共10000条，经过双人标注和质检。'},
            {'question': '可以自定义数据吗', 'answer': '可以的，您可以准备自己的CSV数据文件，然后使用训练脚本重新训练模型。'},
            {'question': '怎么联系客服', 'answer': '您可以通过以下方式联系我们：\n- 邮箱：support@example.com\n- 电话：400-xxx-xxxx\n- 工作时间：周一至周五 9:00-18:00'},
            {'question': '你们是什么公司', 'answer': '我们是广州城建职业学院人工智能技术应用专业的实训项目团队。'},
            {'question': '项目有什么创新', 'answer': '本项目有5个创新点：\n1. 情感联动智能客服\n2. 自动化质检流水线\n3. Git+DVC版本管理\n4. 双平台一键启动\n5. 统一配置管理'},
        ]
        
        self.knowledge_base = pd.DataFrame(default_qa)
        print(f"[信息] 默认知识库初始化完成，共 {len(self.knowledge_base)} 条问答对")
    
    def analyze_sentiment(self, text):
        """
        分析文本情感（简化版，不依赖模型文件）
        
        Args:
            text: 输入文本
            
        Returns:
            dict: 情感分析结果
        """
        # 简单的关键词匹配情感分析
        positive_words = ['好', '棒', '喜欢', '满意', '不错', '优秀', '赞', '推荐', '惊喜', '完美',
                         '很好', '非常好', '太好了', '真的好', '超棒', '最爱', '五星', '好评']
        negative_words = ['差', '烂', '糟糕', '失望', '不好', '垃圾', '讨厌', '后悔', '坑', '骗',
                         '很差', '太差', '不好用', '不满意', '差评', '踩雷', '浪费', '不值']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)
        
        if positive_count > negative_count:
            return {
                'label': 'positive',
                'label_cn': '正面😊',
                'confidence': 0.7 + min(positive_count * 0.05, 0.25)
            }
        elif negative_count > positive_count:
            return {
                'label': 'negative',
                'label_cn': '负面😠',
                'confidence': 0.7 + min(negative_count * 0.05, 0.25)
            }
        else:
            return {
                'label': 'neutral',
                'label_cn': '中性😐',
                'confidence': 0.6
            }
    
    def is_knowledge_question(self, text):
        """
        判断是否为知识性问题
        """
        for keyword in KNOWLEDGE_KEYWORDS:
            if keyword in text:
                return True
        return False
    
    def calculate_similarity(self, text1, text2):
        """
        计算两个文本的相似度（Jaccard相似度）
        """
        words1 = set(jieba.lcut(text1))
        words2 = set(jieba.lcut(text2))
        
        if len(words1 | words2) == 0:
            return 0.0
        
        return len(words1 & words2) / len(words1 | words2)
    
    def find_best_answer(self, question):
        """
        查找最匹配的回答
        
        Args:
            question: 用户问题
            
        Returns:
            tuple: (答案, 相似度)
        """
        if self.knowledge_base is None or len(self.knowledge_base) == 0:
            return None, 0.0
        
        best_similarity = 0.0
        best_answer = None
        
        for _, row in self.knowledge_base.iterrows():
            q = str(row['question'])
            similarity = self.calculate_similarity(question, q)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = row['answer']
        
        return best_answer, best_similarity
    
    def get_comfort_message(self):
        """
        获取安抚话术
        """
        return random.choice(NEGATIVE_COMFORT_MESSAGES)
    
    def chat(self, question):
        """
        对话接口
        
        Args:
            question: 用户问题
            
        Returns:
            dict: 对话结果
        """
        # 1. 情感分析
        sentiment = self.analyze_sentiment(question)
        is_negative = (sentiment['label'] == 'negative' and 
                      sentiment['confidence'] >= NEGATIVE_CONFIDENCE_THRESHOLD)
        
        # 2. 判断是否为知识性问题
        is_knowledge = self.is_knowledge_question(question)
        
        # 3. 查找答案
        answer, similarity = self.find_best_answer(question)
        
        # 4. 构建回复
        response_parts = []
        is_comforted = False
        is_hit = similarity >= QA_SIMILARITY_THRESHOLD
        
        # 如果是高置信度负面情绪且不是知识性问题，先安抚
        if is_negative and not is_knowledge:
            comfort_msg = self.get_comfort_message()
            response_parts.append(comfort_msg)
            is_comforted = True
        
        # 添加答案
        if is_hit and answer:
            response_parts.append(answer)
        else:
            # 未命中
            response_parts.append("抱歉，我暂时无法回答您的问题。我会把您的问题记录下来，后续会不断优化的~")
            # 记录未知问题
            self.unknown_questions.append({
                'question': question,
                'similarity': similarity
            })
        
        # 拼接最终回复
        final_answer = '\n\n'.join(response_parts) if is_comforted else response_parts[0]
        
        return {
            'question': question,
            'answer': final_answer,
            'similarity': round(similarity, 4),
            'is_hit': is_hit,
            'sentiment': sentiment,
            'is_comforted': is_comforted
        }
    
    def save_unknown_questions(self, output_path=None):
        """
        保存未知问题
        """
        if output_path is None:
            output_path = QA_UNKNOWN_QUESTIONS_PATH
        
        if not self.unknown_questions:
            print("[信息] 没有未知问题需要保存")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df = pd.DataFrame(self.unknown_questions)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"[信息] 已保存 {len(self.unknown_questions)} 条未知问题到: {output_path}")


def main():
    """
    测试智能客服
    """
    print("=" * 60)
    print("  情感联动智能客服 - 测试模式")
    print("=" * 60)
    print()
    print("输入 'quit' 或 'exit' 退出")
    print()
    
    chatbot = QAChatbot()
    
    print("\n" + "=" * 60)
    print("  开始对话")
    print("=" * 60)
    print()
    
    while True:
        try:
            question = input("你: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', '再见']:
                print("\n机器人: 再见！祝您生活愉快~")
                break
            
            if not question:
                continue
            
            result = chatbot.chat(question)
            
            print(f"\n机器人: {result['answer']}")
            
            # 显示调试信息
            print(f"\n  [调试] 情感: {result['sentiment']['label_cn']} "
                  f"(置信度: {result['sentiment']['confidence']:.2f})")
            print(f"  [调试] 相似度: {result['similarity']:.4f}")
            print(f"  [调试] 是否命中: {'是' if result['is_hit'] else '否'}")
            print(f"  [调试] 是否安抚: {'是' if result['is_comforted'] else '否'}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n机器人: 对话被中断，再见！")
            break
        except Exception as e:
            print(f"\n[错误] 发生错误: {e}")
    
    # 保存未知问题
    chatbot.save_unknown_questions()
    
    print()
    print("[完成] 对话结束")


if __name__ == '__main__':
    main()
