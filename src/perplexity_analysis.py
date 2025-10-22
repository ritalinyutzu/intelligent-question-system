"""
困惑度（Perplexity）計算詳解
用於評估 LLM 生成題目的品質
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import matplotlib.pyplot as plt

class PerplexityAnalyzer:
    """困惑度分析器（詳細版）"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        支援的模型：
        - gpt2 (124M)
        - gpt2-medium (355M)
        - gpt2-large (774M)
        - gpt2-xl (1.5B)
        - facebook/opt-350m
        - meta-llama/Llama-2-7b-hf (需要權限)
        """
        print(f"載入模型：{model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✓ 模型已載入到 {self.device}")
    
    def calculate_perplexity(self, text: str, verbose: bool = False) -> Dict:
        """
        計算困惑度（詳細版）
        
        困惑度定義：
        PPL = exp(平均 cross-entropy loss)
        
        物理意義：
        - 模型在預測下一個詞時的"困惑程度"
        - PPL = 10 表示模型平均在 10 個詞中猜測
        - 越低 = 文本越符合模型的訓練分布 = 越"標準"
        
        返回：
        {
            'perplexity': float,
            'log_perplexity': float,
            'token_count': int,
            'avg_token_loss': float,
            'token_perplexities': List[float]  # 每個 token 的困惑度
        }
        """
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        # 計算 loss
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits
        
        # 整體困惑度
        perplexity = torch.exp(loss).item()
        log_perplexity = loss.item()
        
        # 每個 token 的困惑度
        token_perplexities = self._calculate_token_perplexities(
            input_ids, logits
        )
        
        results = {
            'perplexity': perplexity,
            'log_perplexity': log_perplexity,
            'token_count': input_ids.shape[1],
            'avg_token_loss': log_perplexity,
            'token_perplexities': token_perplexities
        }
        
        if verbose:
            self._print_detailed_analysis(text, results)
        
        return results
    
    def _calculate_token_perplexities(
        self, 
        input_ids: torch.Tensor, 
        logits: torch.Tensor
    ) -> List[float]:
        """計算每個 token 的困惑度"""
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # 計算每個位置的 cross-entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 轉為困惑度
        token_perplexities = torch.exp(token_losses).cpu().tolist()
        
        return token_perplexities
    
    def _print_detailed_analysis(self, text: str, results: Dict):
        """列印詳細分析"""
        print("\n" + "="*60)
        print("困惑度詳細分析")
        print("="*60)
        print(f"\n文本：{text[:100]}...")
        print(f"\n整體指標：")
        print(f"  困惑度（PPL）：{results['perplexity']:.2f}")
        print(f"  Log 困惑度：{results['log_perplexity']:.4f}")
        print(f"  Token 數量：{results['token_count']}")
        
        # Token 級別分析
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text)
        )
        token_ppls = results['token_perplexities']
        
        print(f"\nToken 困惑度分布：")
        print(f"  平均：{np.mean(token_ppls):.2f}")
        print(f"  最小：{min(token_ppls):.2f}")
        print(f"  最大：{max(token_ppls):.2f}")
        print(f"  標準差：{np.std(token_ppls):.2f}")
        
        # 找出最困惑的 tokens
        sorted_indices = np.argsort(token_ppls)[::-1]
        print(f"\n最困惑的 5 個 tokens：")
        for i in sorted_indices[:5]:
            if i < len(tokens) - 1:
                print(f"  '{tokens[i+1]}': {token_ppls[i]:.2f}")
    
    def compare_questions(self, questions: List[str]) -> pd.DataFrame:
        """比較多個題目的困惑度"""
        results = []
        
        for q in questions:
            ppl_result = self.calculate_perplexity(q)
            results.append({
                'question': q[:50] + '...' if len(q) > 50 else q,
                'perplexity': ppl_result['perplexity'],
                'token_count': ppl_result['token_count'],
                'avg_token_ppl': np.mean(ppl_result['token_perplexities'])
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('perplexity')
        
        return df
    
    def visualize_perplexity_distribution(
        self, 
        good_questions: List[str],
        bad_questions: List[str]
    ):
        """視覺化好題目 vs 壞題目的困惑度分布"""
        good_ppls = [self.calculate_perplexity(q)['perplexity'] 
                     for q in good_questions]
        bad_ppls = [self.calculate_perplexity(q)['perplexity'] 
                    for q in bad_questions]
        
        plt.figure(figsize=(12, 5))
        
        # 直方圖
        plt.subplot(1, 2, 1)
        plt.hist(good_ppls, bins=20, alpha=0.6, label='高品質題目', color='green')
        plt.hist(bad_ppls, bins=20, alpha=0.6, label='低品質題目', color='red')
        plt.xlabel('困惑度', fontsize=12)
        plt.ylabel('數量', fontsize=12)
        plt.title('困惑度分布比較', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot([good_ppls, bad_ppls], labels=['高品質', '低品質'])
        plt.ylabel('困惑度', fontsize=12)
        plt.title('困惑度 Box Plot', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('perplexity_comparison.png', dpi=150)
        print("✓ 圖表已儲存：perplexity_comparison.png")
        
        # 統計分析
        print("\n統計分析：")
        print(f"高品質題目：")
        print(f"  平均困惑度：{np.mean(good_ppls):.2f}")
        print(f"  標準差：{np.std(good_ppls):.2f}")
        print(f"\n低品質題目：")
        print(f"  平均困惑度：{np.mean(bad_ppls):.2f}")
        print(f"  標準差：{np.std(bad_ppls):.2f}")
        
        # 建議閾值
        threshold = np.mean(good_ppls) + 2 * np.std(good_ppls)
        print(f"\n建議困惑度閾值：{threshold:.2f}")


class PerplexityThresholdFinder:
    """困惑度閾值尋找器"""
    
    def __init__(self, analyzer: PerplexityAnalyzer):
        self.analyzer = analyzer
    
    def find_optimal_threshold(
        self,
        questions: List[str],
        quality_labels: List[bool]  # True = 好題目, False = 壞題目
    ) -> Dict:
        """
        找出最佳困惑度閾值
        
        使用 ROC 曲線和 F1 分數
        """
        # 計算所有困惑度
        perplexities = []
        for q in questions:
            ppl = self.analyzer.calculate_perplexity(q)['perplexity']
            perplexities.append(ppl)
        
        perplexities = np.array(perplexities)
        quality_labels = np.array(quality_labels)
        
        # 測試不同閾值
        thresholds = np.linspace(
            perplexities.min(), 
            perplexities.max(), 
            100
        )
        
        results = []
        for threshold in thresholds:
            # 困惑度 < threshold = 預測為好題目
            predictions = perplexities < threshold
            
            # 計算指標
            tp = np.sum((predictions == True) & (quality_labels == True))
            fp = np.sum((predictions == True) & (quality_labels == False))
            tn = np.sum((predictions == False) & (quality_labels == False))
            fn = np.sum((predictions == False) & (quality_labels == True))
            
            accuracy = (tp + tn) / len(quality_labels)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # 找最佳閾值（F1 最大）
        best_result = max(results, key=lambda x: x['f1'])
        
        # 視覺化
        self._plot_threshold_curves(results, best_result)
        
        return best_result
    
    def _plot_threshold_curves(self, results: List[Dict], best: Dict):
        """繪製閾值曲線"""
        thresholds = [r['threshold'] for r in results]
        
        plt.figure(figsize=(12, 5))
        
        # 左圖：各指標 vs 閾值
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, [r['accuracy'] for r in results], label='Accuracy')
        plt.plot(thresholds, [r['precision'] for r in results], label='Precision')
        plt.plot(thresholds, [r['recall'] for r in results], label='Recall')
        plt.plot(thresholds, [r['f1'] for r in results], label='F1', linewidth=2)
        plt.axvline(best['threshold'], color='red', linestyle='--', 
                   label=f"最佳閾值={best['threshold']:.1f}")
        plt.xlabel('困惑度閾值', fontsize=12)
        plt.ylabel('分數', fontsize=12)
        plt.title('閾值 vs 評估指標', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 右圖：F1 分數放大
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, [r['f1'] for r in results], linewidth=2)
        plt.scatter([best['threshold']], [best['f1']], 
                   color='red', s=100, zorder=5)
        plt.xlabel('困惑度閾值', fontsize=12)
        plt.ylabel('F1 分數', fontsize=12)
        plt.title(f"最佳 F1={best['f1']:.3f}", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_optimization.png', dpi=150)
        print("\n✓ 閾值優化圖表已儲存：threshold_optimization.png")


# 使用範例
if __name__ == "__main__":
    import pandas as pd
    
    print("="*60)
    print("困惑度分析範例")
    print("="*60)
    
    # 1. 初始化分析器
    analyzer = PerplexityAnalyzer(model_name="gpt2")
    
    # 2. 分析單個題目
    print("\n1. 分析單個題目...")
    question = "下列何者為憲法保障之基本權利？(A)生存權 (B)工作權 (C)財產權 (D)以上皆是"
    result = analyzer.calculate_perplexity(question, verbose=True)
    
    # 3. 比較多個題目
    print("\n2. 比較多個題目...")
    questions = [
        "下列何者為憲法保障之基本權利？(A)生存權 (B)工作權 (C)財產權 (D)以上皆是",
        "憲法的基本權利包括什麼？選項：A生存權 B工作權 C財產權 D全部",  # 格式較差
        "asdjfk lkjsdf lkjsfd",  # 無意義文本
    ]
    comparison_df = analyzer.compare_questions(questions)
    print("\n比較結果：")
    print(comparison_df)
    
    # 4. 視覺化分析
    print("\n3. 視覺化分析...")
    good_questions = [
        "下列何者為憲法保障之基本權利？(A)生存權 (B)工作權 (C)財產權 (D)以上皆是",
        # ... 更多好題目
    ]
    bad_questions = [
        "憲法基本權利是什麼？A生存權B工作權C財產權D全部",  # 格式不佳
        # ... 更多壞題目
    ]
    analyzer.visualize_perplexity_distribution(good_questions, bad_questions)
    
    # 5. 找最佳閾值
    print("\n4. 尋找最佳困惑度閾值...")
    all_questions = good_questions + bad_questions
    labels = [True] * len(good_questions) + [False] * len(bad_questions)
    
    threshold_finder = PerplexityThresholdFinder(analyzer)
    best_threshold = threshold_finder.find_optimal_threshold(
        all_questions, labels
    )
    
    print("\n最佳閾值結果：")
    print(f"  困惑度閾值：{best_threshold['threshold']:.2f}")
    print(f"  準確率：{best_threshold['accuracy']:.3f}")
    print(f"  精確率：{best_threshold['precision']:.3f}")
    print(f"  召回率：{best_threshold['recall']:.3f}")
    print(f"  F1 分數：{best_threshold['f1']:.3f}")
    
    print("\n" + "="*60)
    print("困惑度分析完成！")
    print("="*60)
