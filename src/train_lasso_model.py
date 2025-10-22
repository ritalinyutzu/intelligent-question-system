"""
Lasso 模型訓練腳本
用於學習題目品質評估模型
"""

import pandas as pd
import numpy as np
from intelligent_question_system import (
    QuestionFeatureExtractor,
    LassoQuestionSelector
)
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class QuestionQualityTrainer:
    """題目品質模型訓練器"""
    
    def __init__(self):
        self.feature_extractor = QuestionFeatureExtractor()
        self.lasso_selector = None
    
    def prepare_training_data(
        self, 
        questions: List[str],
        manual_scores: List[float] = None
    ) -> tuple:
        """
        準備訓練資料
        
        如果沒有人工分數，會使用規則自動生成初始分數
        """
        # 提取特徵
        print("提取題目特徵...")
        features_list = []
        for q in questions:
            features = self.feature_extractor.extract_all_features(q)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # 如果沒有人工分數，使用規則生成
        if manual_scores is None:
            print("沒有人工分數，使用規則自動評分...")
            manual_scores = self._auto_score(features_df)
        
        return features_df, np.array(manual_scores)
    
    def _auto_score(self, features_df: pd.DataFrame) -> List[float]:
        """
        自動評分規則（用於冷啟動）
        
        基於以下規則：
        1. 困惑度低 → 分數高
        2. 結構完整 → 分數高
        3. 長度適中 → 分數高
        4. 獨特性高 → 分數高
        """
        scores = []
        
        for idx, row in features_df.iterrows():
            score = 5.0  # 基礎分
            
            # 困惑度評分（30%權重）
            if row['perplexity'] < 30:
                score += 1.5
            elif row['perplexity'] < 50:
                score += 1.0
            elif row['perplexity'] < 80:
                score += 0.5
            
            # 結構評分（20%權重）
            if row['has_options'] > 0 and row['option_count'] == 4:
                score += 1.0
            
            # 長度評分（15%權重）
            if 50 < row['length'] < 300:
                score += 0.75
            
            # 獨特性評分（15%權重）
            if row['uniqueness'] > 0.7:
                score += 0.75
            
            # 專業性評分（10%權重）
            if row['exam_keyword_density'] > 0.1:
                score += 0.5
            
            # 其他細節（10%權重）
            if row['has_clear_structure'] > 0:
                score += 0.25
            if row['has_question_mark'] > 0:
                score += 0.25
            
            # 限制在 1-10 分
            score = max(1.0, min(10.0, score))
            scores.append(score)
        
        return scores
    
    def train_with_cross_validation(
        self,
        features_df: pd.DataFrame,
        scores: np.ndarray,
        test_alphas: List[float] = None
    ):
        """
        使用交叉驗證訓練，找最佳 alpha
        """
        if test_alphas is None:
            test_alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        print("\n開始交叉驗證尋找最佳 alpha...")
        print(f"測試的 alpha 值：{test_alphas}")
        
        results = []
        
        for alpha in test_alphas:
            selector = LassoQuestionSelector(alpha=alpha)
            
            # 簡單的訓練驗證分割
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)
            
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=42)
            
            # 5折交叉驗證
            cv_scores = cross_val_score(
                lasso, X_scaled, scores, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            
            mean_mse = -cv_scores.mean()
            std_mse = cv_scores.std()
            
            results.append({
                'alpha': alpha,
                'mse': mean_mse,
                'std': std_mse
            })
            
            print(f"  Alpha={alpha:.3f}: MSE={mean_mse:.4f} (±{std_mse:.4f})")
        
        # 選擇最佳 alpha
        best_result = min(results, key=lambda x: x['mse'])
        best_alpha = best_result['alpha']
        
        print(f"\n✓ 最佳 alpha: {best_alpha}")
        
        # 使用最佳 alpha 訓練最終模型
        self.lasso_selector = LassoQuestionSelector(alpha=best_alpha)
        self.lasso_selector.train(features_df, scores)
        
        return self.lasso_selector, results
    
    def analyze_feature_importance(self, features_df: pd.DataFrame):
        """分析特徵重要性"""
        if self.lasso_selector is None:
            raise ValueError("請先訓練模型！")
        
        # 獲取係數
        coefs = self.lasso_selector.lasso.coef_
        feature_names = features_df.columns
        
        # 創建 DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs)
        })
        
        # 排序
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        # 視覺化
        plt.figure(figsize=(10, 8))
        
        # 只顯示非零係數
        non_zero = importance_df[importance_df['abs_coefficient'] > 0.001]
        
        sns.barplot(data=non_zero, y='feature', x='coefficient', palette='coolwarm')
        plt.title('特徵重要性（Lasso 係數）', fontsize=14)
        plt.xlabel('係數值', fontsize=12)
        plt.ylabel('特徵名稱', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        print("\n✓ 特徵重要性圖表已儲存：feature_importance.png")
        
        return importance_df
    
    def visualize_predictions(
        self, 
        features_df: pd.DataFrame, 
        true_scores: np.ndarray
    ):
        """視覺化預測結果"""
        if self.lasso_selector is None:
            raise ValueError("請先訓練模型！")
        
        # 預測
        predicted_scores = self.lasso_selector.predict_quality(features_df)
        
        # 計算指標
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(true_scores, predicted_scores)
        r2 = r2_score(true_scores, predicted_scores)
        
        print(f"\n模型評估：")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # 繪圖
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 散點圖
        axes[0].scatter(true_scores, predicted_scores, alpha=0.6)
        axes[0].plot([true_scores.min(), true_scores.max()], 
                     [true_scores.min(), true_scores.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('實際分數', fontsize=12)
        axes[0].set_ylabel('預測分數', fontsize=12)
        axes[0].set_title(f'預測 vs 實際 (R²={r2:.3f})', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # 殘差圖
        residuals = true_scores - predicted_scores
        axes[1].scatter(predicted_scores, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('預測分數', fontsize=12)
        axes[1].set_ylabel('殘差（實際 - 預測）', fontsize=12)
        axes[1].set_title('殘差分析', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=150)
        print("✓ 預測分析圖表已儲存：prediction_analysis.png")
        
        return {'mse': mse, 'r2': r2}


def load_training_data(file_path: str) -> tuple:
    """
    從 JSON 檔案載入訓練資料
    
    格式：
    [
      {
        "question": "題目內容...",
        "quality_score": 8.5  # 可選，如果沒有會自動評分
      },
      ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item['question'] for item in data]
    
    # 檢查是否有人工分數
    if 'quality_score' in data[0]:
        scores = [item['quality_score'] for item in data]
    else:
        scores = None
    
    return questions, scores


def save_model(lasso_selector, output_path: str):
    """儲存訓練好的模型"""
    import joblib
    joblib.dump(lasso_selector, output_path)
    print(f"✓ 模型已儲存：{output_path}")


def load_model(model_path: str):
    """載入模型"""
    import joblib
    return joblib.load(model_path)


# 主訓練流程
if __name__ == "__main__":
    print("="*60)
    print("Lasso 題目品質模型訓練")
    print("="*60)
    
    # 1. 載入訓練資料
    print("\n1. 載入訓練資料...")
    questions, manual_scores = load_training_data('training_questions.json')
    print(f"   載入 {len(questions)} 個題目")
    
    # 2. 初始化訓練器
    trainer = QuestionQualityTrainer()
    
    # 3. 準備特徵
    print("\n2. 準備訓練資料...")
    features_df, scores = trainer.prepare_training_data(questions, manual_scores)
    print(f"   特徵維度：{features_df.shape}")
    print(f"   分數範圍：{scores.min():.2f} ~ {scores.max():.2f}")
    
    # 4. 交叉驗證訓練
    print("\n3. 訓練模型（交叉驗證）...")
    lasso_selector, cv_results = trainer.train_with_cross_validation(
        features_df, scores
    )
    
    # 5. 分析特徵重要性
    print("\n4. 分析特徵重要性...")
    importance_df = trainer.analyze_feature_importance(features_df)
    print("\nTop 10 重要特徵：")
    print(importance_df.head(10))
    
    # 6. 評估預測效果
    print("\n5. 評估預測效果...")
    metrics = trainer.visualize_predictions(features_df, scores)
    
    # 7. 儲存模型
    print("\n6. 儲存模型...")
    save_model(lasso_selector, 'lasso_quality_model.pkl')
    
    print("\n" + "="*60)
    print("訓練完成！")
    print("="*60)
    
    # 8. 模型總結
    print("\n模型總結：")
    print(f"  - 訓練樣本數：{len(questions)}")
    print(f"  - 原始特徵數：{features_df.shape[1]}")
    print(f"  - 選中特徵數：{len(lasso_selector.selected_features)}")
    print(f"  - 最佳 alpha：{lasso_selector.alpha:.4f}")
    print(f"  - R² 分數：{metrics['r2']:.4f}")
    print(f"  - MSE：{metrics['mse']:.4f}")
