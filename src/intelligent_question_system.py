"""
智慧命題 2.0 系統
使用 Lasso Regression + Perplexity + Low Temperature
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import openai
import anthropic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

class PerplexityCalculator:
    """困惑度計算器"""
    
    def __init__(self, model_name="gpt2"):
        """
        初始化困惑度計算模型
        model_name: 可選 "gpt2", "gpt2-medium", "meta-llama/Llama-2-7b-hf" 等
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # 設定 device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def calculate(self, text: str) -> float:
        """
        計算文本的困惑度
        Perplexity = exp(平均 negative log-likelihood)
        
        困惑度越低 = 模型越"不困惑" = 文本品質越好
        """
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        # 計算 loss
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_batch(self, texts: List[str]) -> List[float]:
        """批次計算困惑度"""
        perplexities = []
        for text in texts:
            ppl = self.calculate(text)
            perplexities.append(ppl)
        return perplexities


class QuestionFeatureExtractor:
    """題目特徵工程"""
    
    def __init__(self, reference_questions: List[str] = None):
        """
        reference_questions: 參考題庫（用於計算相似度）
        """
        self.reference_questions = reference_questions or []
        self.perplexity_calc = PerplexityCalculator()
    
    def extract_all_features(self, question: str, answer: str = None) -> Dict[str, float]:
        """
        提取所有特徵
        返回特徵字典
        """
        features = {}
        
        # 1. 基礎特徵
        features.update(self._basic_features(question))
        
        # 2. 困惑度特徵
        features['perplexity'] = self.perplexity_calc.calculate(question)
        
        # 3. 語義特徵
        features.update(self._semantic_features(question))
        
        # 4. 結構特徵
        features.update(self._structure_features(question))
        
        # 5. 答案相關特徵（如果有答案）
        if answer:
            features.update(self._answer_features(question, answer))
        
        # 6. 多樣性特徵
        features['uniqueness'] = self._calculate_uniqueness(question)
        
        return features
    
    def _basic_features(self, question: str) -> Dict[str, float]:
        """基礎統計特徵"""
        return {
            'length': len(question),
            'word_count': len(question.split()),
            'avg_word_length': np.mean([len(w) for w in question.split()]),
            'char_count': len(question.replace(' ', '')),
            'has_question_mark': 1.0 if '?' in question else 0.0,
            'punctuation_count': sum(1 for c in question if c in ',.!?;:'),
        }
    
    def _semantic_features(self, question: str) -> Dict[str, float]:
        """語義相關特徵"""
        # 檢查是否包含考試常見詞彙
        exam_keywords = ['下列', '何者', '正確', '錯誤', '選項', '最', '不', '是']
        keyword_count = sum(1 for kw in exam_keywords if kw in question)
        
        # 檢查專業術語密度（可根據科目自定義）
        professional_terms = self._count_professional_terms(question)
        
        return {
            'exam_keyword_density': keyword_count / max(len(question.split()), 1),
            'professional_term_count': professional_terms,
        }
    
    def _structure_features(self, question: str) -> Dict[str, float]:
        """結構特徵"""
        # 檢查選項格式
        has_options = bool(re.search(r'[A-D][\.)、]', question))
        option_count = len(re.findall(r'[A-D][\.)、]', question))
        
        # 檢查是否有題幹和選項分離
        has_clear_structure = '\n' in question or '：' in question
        
        return {
            'has_options': 1.0 if has_options else 0.0,
            'option_count': float(option_count),
            'has_clear_structure': 1.0 if has_clear_structure else 0.0,
        }
    
    def _answer_features(self, question: str, answer: str) -> Dict[str, float]:
        """答案品質特徵"""
        return {
            'answer_length': len(answer),
            'answer_word_count': len(answer.split()),
            'has_explanation': 1.0 if len(answer) > 50 else 0.0,
        }
    
    def _count_professional_terms(self, text: str) -> int:
        """計算專業術語數量（需根據科目自定義）"""
        # 這裡是範例，實際應根據不同考試科目建立專業詞庫
        sample_terms = ['法律', '憲法', '行政', '刑法', '民法', '訴訟', '權利', '義務']
        count = sum(1 for term in sample_terms if term in text)
        return count
    
    def _calculate_uniqueness(self, question: str) -> float:
        """
        計算題目獨特性（與現有題庫的差異度）
        使用簡單的 Jaccard 距離
        """
        if not self.reference_questions:
            return 1.0
        
        # 將題目轉為字詞集合
        q_set = set(question.split())
        
        # 計算與所有參考題目的最大相似度
        max_similarity = 0.0
        for ref_q in self.reference_questions:
            ref_set = set(ref_q.split())
            
            # Jaccard 相似度
            intersection = len(q_set & ref_set)
            union = len(q_set | ref_set)
            similarity = intersection / union if union > 0 else 0
            
            max_similarity = max(max_similarity, similarity)
        
        # 獨特性 = 1 - 最大相似度
        uniqueness = 1.0 - max_similarity
        return uniqueness


class LassoQuestionSelector:
    """使用 Lasso 回歸進行題目篩選"""
    
    def __init__(self, alpha: float = 0.1):
        """
        alpha: Lasso 懲罰係數
        - 越大：越多特徵被設為 0（更嚴格的特徵選擇）
        - 越小：保留更多特徵
        建議範圍：0.01 ~ 0.5
        """
        self.alpha = alpha
        self.lasso = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.selected_features = None
    
    def train(self, features: pd.DataFrame, quality_scores: np.ndarray):
        """
        訓練 Lasso 模型
        
        features: 特徵矩陣（每行一個題目，每列一個特徵）
        quality_scores: 品質分數（1-10分，人工標註或規則計算）
        """
        self.feature_names = features.columns.tolist()
        
        # 標準化特徵
        X_scaled = self.scaler.fit_transform(features)
        
        # 使用交叉驗證選擇最佳 alpha（可選）
        if self.alpha is None:
            alphas = np.logspace(-3, 1, 50)
            lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
            lasso_cv.fit(X_scaled, quality_scores)
            self.alpha = lasso_cv.alpha_
            print(f"最佳 alpha: {self.alpha:.4f}")
        
        # 訓練 Lasso 模型
        self.lasso = Lasso(alpha=self.alpha, random_state=42)
        self.lasso.fit(X_scaled, quality_scores)
        
        # 記錄被選中的特徵
        self.selected_features = [
            name for name, coef in zip(self.feature_names, self.lasso.coef_) 
            if abs(coef) > 1e-5
        ]
        
        print(f"\n特徵選擇結果：")
        print(f"原始特徵數：{len(self.feature_names)}")
        print(f"選中特徵數：{len(self.selected_features)}")
        print(f"被選中的特徵：{self.selected_features}")
        
        # 顯示特徵重要性
        self._print_feature_importance()
        
        return self
    
    def predict_quality(self, features: pd.DataFrame) -> np.ndarray:
        """預測題目品質分數"""
        X_scaled = self.scaler.transform(features)
        scores = self.lasso.predict(X_scaled)
        return scores
    
    def select_top_questions(
        self, 
        questions: List[str], 
        features: pd.DataFrame, 
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        選擇品質最高的 top_k 個題目
        
        返回：[(題目, 品質分數), ...]
        """
        scores = self.predict_quality(features)
        
        # 排序
        ranked_indices = np.argsort(scores)[::-1]  # 降序
        
        # 返回 top_k
        top_questions = [
            (questions[i], scores[i]) 
            for i in ranked_indices[:top_k]
        ]
        
        return top_questions
    
    def _print_feature_importance(self):
        """顯示特徵重要性"""
        importance = list(zip(self.feature_names, self.lasso.coef_))
        importance = sorted(importance, key=lambda x: abs(x[1]), reverse=True)
        
        print("\n特徵重要性（係數絕對值排序）：")
        for name, coef in importance[:10]:  # 只顯示前 10 個
            print(f"  {name:30s}: {coef:8.4f}")


class LowTempQuestionGenerator:
    """低溫題目生成器"""
    
    def __init__(
        self, 
        provider: str = "openai",  # "openai", "anthropic", "gemini"
        model: str = None,
        api_key: str = None
    ):
        """
        provider: API 提供商
        model: 模型名稱
        api_key: API 金鑰
        """
        self.provider = provider
        self.api_key = api_key
        
        # 設定模型
        if provider == "openai":
            self.model = model or "gpt-4"
            openai.api_key = api_key
        elif provider == "anthropic":
            self.model = model or "claude-3-sonnet-20240229"
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(
        self, 
        topic: str, 
        difficulty: str = "中等",
        count: int = 1,
        temperature: float = 0.2,  # 低溫！
        additional_requirements: str = ""
    ) -> List[Dict[str, str]]:
        """
        生成題目
        
        temperature: 溫度參數
        - 0.0-0.3: 非常穩定，貼近訓練數據
        - 0.4-0.6: 平衡
        - 0.7-1.0: 更有創意但可能偏離標準
        """
        prompt = self._build_prompt(topic, difficulty, additional_requirements)
        
        questions = []
        
        if self.provider == "openai":
            questions = self._generate_openai(prompt, count, temperature)
        elif self.provider == "anthropic":
            questions = self._generate_anthropic(prompt, count, temperature)
        
        return questions
    
    def _build_prompt(self, topic: str, difficulty: str, requirements: str) -> str:
        """建立 prompt"""
        prompt = f"""你是一個專業的考試命題專家。請根據以下要求生成高品質的選擇題。

主題：{topic}
難度：{difficulty}
額外要求：{requirements}

請生成一道選擇題，包含：
1. 題幹（清晰、專業、無歧義）
2. 4個選項（A、B、C、D）
3. 正確答案
4. 詳細解析

要求：
- 題目必須符合考試標準格式
- 選項要有合理的干擾性
- 避免明顯的提示或線索
- 確保答案絕對正確
- 解析要詳細且有教育價值

輸出格式（JSON）：
{{
  "question": "題幹內容",
  "options": {{
    "A": "選項A",
    "B": "選項B",
    "C": "選項C",
    "D": "選項D"
  }},
  "correct_answer": "A",
  "explanation": "詳細解析"
}}
"""
        return prompt
    
    def _generate_openai(self, prompt: str, count: int, temperature: float) -> List[Dict]:
        """使用 OpenAI API 生成"""
        questions = []
        
        for _ in range(count):
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            question = json.loads(content)
            questions.append(question)
        
        return questions
    
    def _generate_anthropic(self, prompt: str, count: int, temperature: float) -> List[Dict]:
        """使用 Anthropic API 生成"""
        questions = []
        
        for _ in range(count):
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = message.content[0].text
            # 嘗試解析 JSON
            try:
                question = json.loads(content)
            except:
                # 如果不是 JSON，嘗試正則提取
                question = self._parse_text_response(content)
            
            questions.append(question)
        
        return questions
    
    def _parse_text_response(self, text: str) -> Dict:
        """從文本中提取題目結構"""
        # 簡化版，實際需要更完善的解析
        return {
            "question": text,
            "options": {},
            "correct_answer": "",
            "explanation": ""
        }


class IntelligentQuestionSystem:
    """智慧命題 2.0 完整系統"""
    
    def __init__(
        self,
        api_provider: str = "openai",
        api_key: str = None,
        lasso_alpha: float = 0.1,
        perplexity_threshold: float = 50.0,
        temperature: float = 0.2
    ):
        """
        api_provider: LLM API 提供商
        api_key: API 金鑰
        lasso_alpha: Lasso 懲罰係數
        perplexity_threshold: 困惑度閾值
        temperature: 生成溫度
        """
        self.generator = LowTempQuestionGenerator(api_provider, api_key=api_key)
        self.feature_extractor = QuestionFeatureExtractor()
        self.lasso_selector = LassoQuestionSelector(alpha=lasso_alpha)
        self.perplexity_calc = PerplexityCalculator()
        
        self.perplexity_threshold = perplexity_threshold
        self.temperature = temperature
        
        self.is_trained = False
    
    def train_quality_model(
        self, 
        training_questions: List[str],
        quality_scores: List[float]
    ):
        """
        訓練品質評估模型
        
        training_questions: 訓練題目列表
        quality_scores: 對應的品質分數（1-10）
        """
        print("開始訓練品質評估模型...")
        
        # 提取特徵
        print("提取特徵中...")
        features_list = []
        for q in training_questions:
            features = self.feature_extractor.extract_all_features(q)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # 訓練 Lasso
        self.lasso_selector.train(features_df, np.array(quality_scores))
        
        self.is_trained = True
        print("訓練完成！")
    
    def generate_and_filter(
        self,
        topic: str,
        difficulty: str = "中等",
        target_count: int = 20,
        candidate_multiplier: int = 5
    ) -> List[Dict]:
        """
        完整流程：生成 → 困惑度篩選 → Lasso 排序 → 返回最佳題目
        
        target_count: 最終要保留的題目數量
        candidate_multiplier: 候選題目倍數（生成 target_count * multiplier 個候選）
        """
        if not self.is_trained:
            raise ValueError("請先使用 train_quality_model() 訓練模型！")
        
        print(f"\n{'='*60}")
        print(f"開始生成題目：{topic} / 難度：{difficulty}")
        print(f"目標題目數：{target_count}")
        print(f"{'='*60}\n")
        
        # Step 1: 大量生成候選題目
        candidate_count = target_count * candidate_multiplier
        print(f"Step 1: 生成 {candidate_count} 個候選題目（低溫 {self.temperature}）...")
        
        candidates = self.generator.generate(
            topic=topic,
            difficulty=difficulty,
            count=candidate_count,
            temperature=self.temperature
        )
        
        print(f"✓ 生成完成：{len(candidates)} 個候選題目\n")
        
        # Step 2: 困惑度篩選
        print(f"Step 2: 困惑度篩選（閾值 < {self.perplexity_threshold}）...")
        
        passed_perplexity = []
        perplexities = []
        
        for candidate in candidates:
            question_text = self._format_question(candidate)
            ppl = self.perplexity_calc.calculate(question_text)
            perplexities.append(ppl)
            
            if ppl < self.perplexity_threshold:
                passed_perplexity.append(candidate)
        
        print(f"✓ 通過困惑度篩選：{len(passed_perplexity)}/{len(candidates)} 個")
        print(f"  平均困惑度：{np.mean(perplexities):.2f}")
        print(f"  困惑度範圍：{min(perplexities):.2f} ~ {max(perplexities):.2f}\n")
        
        if len(passed_perplexity) < target_count:
            print(f"⚠ 警告：通過篩選的題目少於目標數量，可能需要調整困惑度閾值")
        
        # Step 3: Lasso 特徵選擇與品質預測
        print(f"Step 3: Lasso 品質評估與排序...")
        
        # 提取特徵
        features_list = []
        question_texts = []
        
        for candidate in passed_perplexity:
            question_text = self._format_question(candidate)
            question_texts.append(question_text)
            
            features = self.feature_extractor.extract_all_features(question_text)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # 預測品質
        quality_scores = self.lasso_selector.predict_quality(features_df)
        
        # 排序
        ranked_indices = np.argsort(quality_scores)[::-1]
        
        print(f"✓ 品質評分完成")
        print(f"  平均品質分數：{np.mean(quality_scores):.2f}")
        print(f"  分數範圍：{min(quality_scores):.2f} ~ {max(quality_scores):.2f}\n")
        
        # Step 4: 返回 Top K
        print(f"Step 4: 選擇品質最高的 {target_count} 個題目\n")
        
        final_questions = []
        for i in ranked_indices[:target_count]:
            question = passed_perplexity[i]
            question['quality_score'] = quality_scores[i]
            question['perplexity'] = perplexities[i]
            final_questions.append(question)
        
        print(f"{'='*60}")
        print(f"完成！最終產出 {len(final_questions)} 個高品質題目")
        print(f"{'='*60}\n")
        
        return final_questions
    
    def _format_question(self, question_dict: Dict) -> str:
        """將題目字典格式化為文本"""
        text = question_dict.get('question', '')
        
        options = question_dict.get('options', {})
        for key, value in options.items():
            text += f"\n{key}. {value}"
        
        return text
    
    def export_results(self, questions: List[Dict], output_file: str):
        """匯出結果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 題目已匯出至：{output_file}")


# 使用範例
if __name__ == "__main__":
    # 初始化系統
    system = IntelligentQuestionSystem(
        api_provider="openai",
        api_key="your-api-key-here",
        lasso_alpha=0.1,
        perplexity_threshold=50.0,
        temperature=0.2
    )
    
    # 訓練模型（使用現有題目）
    training_questions = [
        "下列何者為憲法保障之基本權利？(A)生存權 (B)工作權 (C)財產權 (D)以上皆是",
        # ... 更多訓練題目
    ]
    quality_scores = [8.5, 7.0, 9.0, 6.5]  # 人工標註的品質分數
    
    system.train_quality_model(training_questions, quality_scores)
    
    # 生成新題目
    results = system.generate_and_filter(
        topic="憲法基本權利",
        difficulty="中等",
        target_count=20
    )
    
    # 匯出
    system.export_results(results, "output_questions.json")
