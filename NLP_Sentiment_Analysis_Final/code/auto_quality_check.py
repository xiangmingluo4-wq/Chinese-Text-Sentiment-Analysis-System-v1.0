# -*- coding: utf-8 -*-
"""
自动化质检脚本
中文文本情感分析系统 - 训练前置质检流水线
自动检查数据质量，生成质检报告
"""

import os
import sys
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# 设置随机种子
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class QualityChecker:
    """
    自动化质检类
    """
    
    def __init__(self, data_path=None):
        """
        初始化质检器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path or SENTIMENT_DATA_PATH
        self.df = None
        self.results = {}
        self.passed = False
    
    def load_data(self):
        """
        加载数据
        """
        print(f"[质检] 正在加载数据: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"[错误] 数据文件不存在: {self.data_path}")
            return False
        
        self.df = pd.read_csv(self.data_path)
        print(f"[质检] 数据加载完成，共 {len(self.df)} 条")
        
        return True
    
    def check_data_volume(self):
        """
        检查数据量
        """
        print("\n[质检1/6] 检查数据量...")
        
        total = len(self.df)
        min_required = 1000  # 最低要求
        
        self.results['data_volume'] = {
            'total': total,
            'min_required': min_required,
            'passed': total >= min_required
        }
        
        if total >= min_required:
            print(f"  ✅ 数据量: {total} 条 (满足要求 ≥{min_required})")
        else:
            print(f"  ❌ 数据量: {total} 条 (不满足要求 ≥{min_required})")
        
        return self.results['data_volume']['passed']
    
    def check_label_completeness(self):
        """
        检查标签完整性
        """
        print("\n[质检2/6] 检查标签完整性...")
        
        required_columns = ['id', 'text', 'sentiment', 'annotator1', 'annotator2']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        # 检查空值
        null_counts = {}
        for col in required_columns:
            if col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                null_counts[col] = null_count
        
        total_nulls = sum(null_counts.values())
        
        self.results['label_completeness'] = {
            'missing_columns': missing_columns,
            'null_counts': null_counts,
            'total_nulls': total_nulls,
            'passed': len(missing_columns) == 0 and total_nulls == 0
        }
        
        if len(missing_columns) == 0 and total_nulls == 0:
            print(f"  ✅ 标签完整，无缺失值")
        else:
            if missing_columns:
                print(f"  ❌ 缺少列: {missing_columns}")
            if total_nulls > 0:
                print(f"  ❌ 空值数量: {total_nulls}")
        
        return self.results['label_completeness']['passed']
    
    def check_label_distribution(self):
        """
        检查标签分布
        """
        print("\n[质检3/6] 检查情感标签分布...")
        
        if 'sentiment' not in self.df.columns:
            self.results['label_distribution'] = {'passed': False, 'reason': '缺少sentiment列'}
            print("  ❌ 缺少sentiment列")
            return False
        
        label_counts = self.df['sentiment'].value_counts()
        total = len(self.df)
        
        # 检查是否包含所有三个类别
        missing_labels = [l for l in SENTIMENT_LABELS if l not in label_counts.index]
        
        # 检查分布是否过于不平衡（某类别<5%）
        min_ratio = 0.05
        imbalanced_labels = []
        for label in SENTIMENT_LABELS:
            count = label_counts.get(label, 0)
            ratio = count / total if total > 0 else 0
            if ratio < min_ratio:
                imbalanced_labels.append((label, ratio))
        
        self.results['label_distribution'] = {
            'label_counts': label_counts.to_dict(),
            'missing_labels': missing_labels,
            'imbalanced_labels': imbalanced_labels,
            'passed': len(missing_labels) == 0 and len(imbalanced_labels) == 0
        }
        
        if len(missing_labels) == 0 and len(imbalanced_labels) == 0:
            print(f"  ✅ 标签分布正常")
            for label in SENTIMENT_LABELS:
                count = label_counts.get(label, 0)
                ratio = count / total * 100
                print(f"     {SENTIMENT_MAP[label]}: {count} 条 ({ratio:.1f}%)")
        else:
            if missing_labels:
                print(f"  ❌ 缺少类别: {missing_labels}")
            if imbalanced_labels:
                print(f"  ⚠️  类别不平衡:")
                for label, ratio in imbalanced_labels:
                    print(f"     {SENTIMENT_MAP[label]}: {ratio*100:.1f}%")
        
        return self.results['label_distribution']['passed']
    
    def check_annotation_agreement(self):
        """
        检查标注一致性（Kappa系数）
        """
        print("\n[质检4/6] 检查标注一致性...")
        
        if 'annotator1' not in self.df.columns or 'annotator2' not in self.df.columns:
            self.results['annotation_agreement'] = {'passed': False, 'reason': '缺少标注列'}
            print("  ❌ 缺少annotator1或annotator2列")
            return False
        
        # 计算一致率
        agree_count = (self.df['annotator1'] == self.df['annotator2']).sum()
        agree_rate = agree_count / len(self.df)
        
        # 计算Kappa系数
        try:
            kappa = cohen_kappa_score(self.df['annotator1'], self.df['annotator2'])
        except:
            kappa = 0.0
        
        # 判定标准
        kappa_threshold = KAPPA_THRESHOLD
        agree_rate_threshold = 0.8
        
        passed = kappa >= kappa_threshold and agree_rate >= agree_rate_threshold
        
        self.results['annotation_agreement'] = {
            'agree_count': int(agree_count),
            'agree_rate': round(agree_rate, 4),
            'kappa': round(kappa, 4),
            'kappa_threshold': kappa_threshold,
            'passed': passed
        }
        
        if passed:
            print(f"  ✅ 标注一致性达标")
            print(f"     一致率: {agree_rate*100:.1f}%")
            print(f"     Kappa系数: {kappa:.4f} (阈值: {kappa_threshold})")
        else:
            print(f"  ❌ 标注一致性不达标")
            print(f"     一致率: {agree_rate*100:.1f}% (阈值: {agree_rate_threshold*100:.0f}%)")
            print(f"     Kappa系数: {kappa:.4f} (阈值: {kappa_threshold})")
        
        return passed
    
    def check_text_quality(self):
        """
        检查文本质量
        """
        print("\n[质检5/6] 检查文本质量...")
        
        if 'text' not in self.df.columns:
            self.results['text_quality'] = {'passed': False, 'reason': '缺少text列'}
            print("  ❌ 缺少text列")
            return False
        
        # 计算文本长度
        text_lengths = self.df['text'].astype(str).str.len()
        
        # 检查过短文本
        too_short = (text_lengths < MIN_TEXT_LENGTH).sum()
        too_short_ratio = too_short / len(self.df)
        
        # 检查过长文本
        too_long = (text_lengths > MAX_TEXT_LENGTH).sum()
        too_long_ratio = too_long / len(self.df)
        
        # 检查重复文本
        duplicates = self.df['text'].duplicated().sum()
        duplicate_ratio = duplicates / len(self.df)
        
        # 判定标准
        anomaly_threshold = ANOMALY_RATE_THRESHOLD
        
        total_anomalies = too_short + too_long + duplicates
        total_anomaly_ratio = total_anomalies / len(self.df)
        
        passed = total_anomaly_ratio <= anomaly_threshold
        
        self.results['text_quality'] = {
            'avg_length': round(text_lengths.mean(), 1),
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'too_short': int(too_short),
            'too_short_ratio': round(too_short_ratio, 4),
            'too_long': int(too_long),
            'too_long_ratio': round(too_long_ratio, 4),
            'duplicates': int(duplicates),
            'duplicate_ratio': round(duplicate_ratio, 4),
            'total_anomalies': int(total_anomalies),
            'total_anomaly_ratio': round(total_anomaly_ratio, 4),
            'passed': passed
        }
        
        if passed:
            print(f"  ✅ 文本质量达标")
            print(f"     平均长度: {text_lengths.mean():.1f} 字符")
            print(f"     异常率: {total_anomaly_ratio*100:.2f}% (阈值: {anomaly_threshold*100:.0f}%)")
        else:
            print(f"  ❌ 文本质量不达标")
            print(f"     过短文本: {too_short} 条 ({too_short_ratio*100:.2f}%)")
            print(f"     过长文本: {too_long} 条 ({too_long_ratio*100:.2f}%)")
            print(f"     重复文本: {duplicates} 条 ({duplicate_ratio*100:.2f}%)")
            print(f"     总异常率: {total_anomaly_ratio*100:.2f}% (阈值: {anomaly_threshold*100:.0f}%)")
        
        return passed
    
    def check_domain_coverage(self):
        """
        检查领域覆盖
        """
        print("\n[质检6/6] 检查领域覆盖...")
        
        if 'domain' not in self.df.columns:
            self.results['domain_coverage'] = {'passed': False, 'reason': '缺少domain列'}
            print("  ⚠️  缺少domain列，跳过检查")
            return True  # 非强制项
        
        domain_counts = self.df['domain'].value_counts()
        covered_domains = domain_counts.index.tolist()
        missing_domains = [d for d in DOMAINS if d not in covered_domains]
        
        # 检查每个领域的最小样本量
        min_per_domain = 100
        insufficient_domains = []
        for domain in DOMAINS:
            count = domain_counts.get(domain, 0)
            if count < min_per_domain:
                insufficient_domains.append((domain, count))
        
        passed = len(missing_domains) == 0 and len(insufficient_domains) == 0
        
        self.results['domain_coverage'] = {
            'total_domains': len(DOMAINS),
            'covered_domains': len(covered_domains),
            'missing_domains': missing_domains,
            'insufficient_domains': insufficient_domains,
            'passed': passed
        }
        
        if passed:
            print(f"  ✅ 领域覆盖完整")
            print(f"     覆盖 {len(covered_domains)}/{len(DOMAINS)} 个领域")
        else:
            if missing_domains:
                print(f"  ⚠️  缺少领域: {missing_domains}")
            if insufficient_domains:
                print(f"  ⚠️  样本不足的领域:")
                for domain, count in insufficient_domains:
                    print(f"     {domain}: {count} 条 (阈值: {min_per_domain})")
        
        return passed  # 领域覆盖非强制项
    
    def run_full_check(self):
        """
        执行完整质检流程
        """
        print("=" * 70)
        print("  中文文本情感分析系统 - 自动化质检")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        # 加载数据
        if not self.load_data():
            self.passed = False
            return False
        
        # 执行各项检查
        checks = [
            self.check_data_volume(),
            self.check_label_completeness(),
            self.check_label_distribution(),
            self.check_annotation_agreement(),
            self.check_text_quality(),
            self.check_domain_coverage(),
        ]
        
        # 总体判定（前5项必须通过，第6项可选）
        mandatory_passed = all(checks[:5])
        self.passed = mandatory_passed
        
        total_time = time.time() - start_time
        
        # 打印总结
        print("\n" + "=" * 70)
        print("  质检总结")
        print("=" * 70)
        print()
        
        if self.passed:
            print("  ✅ 质检通过！数据可以进入训练流程")
        else:
            print("  ❌ 质检未通过！请修复后再进行训练")
        
        print(f"\n  总耗时: {total_time:.2f} 秒")
        print()
        
        # 生成报告
        self.generate_report()
        
        return self.passed
    
    def generate_report(self, output_dir=None):
        """
        生成质检报告
        """
        if output_dir is None:
            output_dir = os.path.join(DOCS_DIR, '质检报告')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, f'质检报告_{timestamp}.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 数据质检报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**数据文件**: {self.data_path}\n\n")
            f.write(f"**质检结果**: {'✅ 通过' if self.passed else '❌ 未通过'}\n\n")
            
            f.write("---\n\n")
            
            # 1. 数据量检查
            dv = self.results.get('data_volume', {})
            f.write("## 1. 数据量检查\n\n")
            f.write(f"- 总数据量: {dv.get('total', 0)} 条\n")
            f.write(f"- 最低要求: {dv.get('min_required', 0)} 条\n")
            f.write(f"- 结果: {'✅ 通过' if dv.get('passed') else '❌ 未通过'}\n\n")
            
            # 2. 标签完整性
            lc = self.results.get('label_completeness', {})
            f.write("## 2. 标签完整性检查\n\n")
            if lc.get('missing_columns'):
                f.write(f"- 缺少列: {lc['missing_columns']}\n")
            f.write(f"- 空值总数: {lc.get('total_nulls', 0)}\n")
            f.write(f"- 结果: {'✅ 通过' if lc.get('passed') else '❌ 未通过'}\n\n")
            
            # 3. 标签分布
            ld = self.results.get('label_distribution', {})
            f.write("## 3. 情感标签分布\n\n")
            label_counts = ld.get('label_counts', {})
            total = dv.get('total', 1)
            for label in SENTIMENT_LABELS:
                count = label_counts.get(label, 0)
                ratio = count / total * 100 if total > 0 else 0
                f.write(f"- {SENTIMENT_MAP[label]}: {count} 条 ({ratio:.1f}%)\n")
            f.write(f"- 结果: {'✅ 通过' if ld.get('passed') else '⚠️ 警告'}\n\n")
            
            # 4. 标注一致性
            aa = self.results.get('annotation_agreement', {})
            f.write("## 4. 标注一致性检查\n\n")
            f.write(f"- 一致数量: {aa.get('agree_count', 0)} 条\n")
            f.write(f"- 一致率: {aa.get('agree_rate', 0)*100:.1f}%\n")
            f.write(f"- Kappa系数: {aa.get('kappa', 0):.4f}\n")
            f.write(f"- Kappa阈值: {aa.get('kappa_threshold', 0)}\n")
            f.write(f"- 结果: {'✅ 通过' if aa.get('passed') else '❌ 未通过'}\n\n")
            
            # 5. 文本质量
            tq = self.results.get('text_quality', {})
            f.write("## 5. 文本质量检查\n\n")
            f.write(f"- 平均长度: {tq.get('avg_length', 0)} 字符\n")
            f.write(f"- 最短长度: {tq.get('min_length', 0)} 字符\n")
            f.write(f"- 最长长度: {tq.get('max_length', 0)} 字符\n")
            f.write(f"- 过短文本: {tq.get('too_short', 0)} 条 ({tq.get('too_short_ratio', 0)*100:.2f}%)\n")
            f.write(f"- 过长文本: {tq.get('too_long', 0)} 条 ({tq.get('too_long_ratio', 0)*100:.2f}%)\n")
            f.write(f"- 重复文本: {tq.get('duplicates', 0)} 条 ({tq.get('duplicate_ratio', 0)*100:.2f}%)\n")
            f.write(f"- 总异常率: {tq.get('total_anomaly_ratio', 0)*100:.2f}%\n")
            f.write(f"- 结果: {'✅ 通过' if tq.get('passed') else '❌ 未通过'}\n\n")
            
            # 6. 领域覆盖
            dc = self.results.get('domain_coverage', {})
            f.write("## 6. 领域覆盖检查\n\n")
            f.write(f"- 总领域数: {dc.get('total_domains', 0)}\n")
            f.write(f"- 已覆盖: {dc.get('covered_domains', 0)}\n")
            if dc.get('missing_domains'):
                f.write(f"- 缺少领域: {dc['missing_domains']}\n")
            f.write(f"- 结果: {'✅ 通过' if dc.get('passed') else '⚠️ 警告'}\n\n")
            
            # 总结
            f.write("---\n\n")
            f.write("## 总结\n\n")
            if self.passed:
                f.write("✅ **质检通过**，数据可以进入训练流程。\n")
            else:
                f.write("❌ **质检未通过**，请修复以下问题后再进行训练：\n\n")
                if not dv.get('passed'):
                    f.write("- [ ] 数据量不足\n")
                if not lc.get('passed'):
                    f.write("- [ ] 标签不完整\n")
                if not ld.get('passed'):
                    f.write("- [ ] 标签分布异常\n")
                if not aa.get('passed'):
                    f.write("- [ ] 标注一致性不达标\n")
                if not tq.get('passed'):
                    f.write("- [ ] 文本质量不达标\n")
        
        print(f"\n[质检] 报告已生成: {report_path}")
        
        return report_path


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='自动化质检脚本')
    parser.add_argument('--data', '-d', type=str, default=None, 
                        help='数据文件路径')
    parser.add_argument('--strict', '-s', action='store_true',
                        help='严格模式，质检失败直接退出')
    
    args = parser.parse_args()
    
    # 创建质检器
    checker = QualityChecker(data_path=args.data)
    
    # 执行质检
    passed = checker.run_full_check()
    
    # 严格模式下失败退出
    if args.strict and not passed:
        print("\n[错误] 严格模式下质检失败，程序退出")
        sys.exit(1)
    
    return passed


if __name__ == '__main__':
    main()
