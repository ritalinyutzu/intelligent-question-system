"""
智慧命題 2.0 - 完整使用範例
從零開始到產出高品質題目
"""

from intelligent_question_system import IntelligentQuestionSystem
from train_lasso_model import QuestionQualityTrainer, save_model, load_model
from perplexity_analysis import PerplexityAnalyzer, PerplexityThresholdFinder
import json

# ============================================================================
# Phase 1: 準備訓練資料
# ============================================================================

def prepare_initial_training_data():
    """
    準備初始訓練資料
    
    方式 1：使用現有題庫
    方式 2：手動標註一小批題目（推薦至少 50-100 題）
    """
    # 範例：從現有題庫載入
    existing_questions = [
        {
            "question": "下列何者為憲法保障之基本權利？(A)生存權 (B)工作權 (C)財產權 (D)以上皆是",
            "quality_score": 9.0  # 人工評分 1-10
        },
        {
            "question": "依憲法規定，人民有工作之權利與義務，下列敘述何者正確？(A)國家應保障人民工作機會 (B)人民有選擇職業之自由 (C)國家應實施職業教育 (D)以上皆是",
            "quality_score": 8.5
        },
        {
            "question": "憲法的權利有哪些？A生存權B工作權C財產權D全部",  # 格式不佳
            "quality_score": 4.0
        },
        # ... 新增更多題目
    ]
    
    # 儲存為 JSON
    with open('training_questions.json', 'w', encoding='utf-8') as f:
        json.dump(existing_questions, f, ensure_ascii=False, indent=2)
    
    print("✓ 訓練資料已準備：training_questions.json")
    return existing_questions


# ============================================================================
# Phase 2: 訓練 Lasso 品質模型
# ============================================================================

def train_quality_model():
    """訓練品質評估模型"""
    print("\n" + "="*60)
    print("Phase 2: 訓練 Lasso 品質模型")
    print("="*60)
    
    # 載入訓練資料
    with open('training_questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item['question'] for item in data]
    scores = [item['quality_score'] for item in data]
    
    # 初始化訓練器
    trainer = QuestionQualityTrainer()
    
    # 準備資料
    features_df, scores_array = trainer.prepare_training_data(questions, scores)
    
    # 交叉驗證訓練
    lasso_selector, cv_results = trainer.train_with_cross_validation(
        features_df, scores_array
    )
    
    # 分析特徵
    importance_df = trainer.analyze_feature_importance(features_df)
    
    # 評估效果
    metrics = trainer.visualize_predictions(features_df, scores_array)
    
    # 儲存模型
    save_model(lasso_selector, 'lasso_quality_model.pkl')
    
    print("\n✓ Lasso 模型訓練完成！")
    return lasso_selector


# ============================================================================
# Phase 3: 分析困惑度並設定閾值
# ============================================================================

def analyze_and_set_perplexity_threshold():
    """分析困惑度並找出最佳閾值"""
    print("\n" + "="*60)
    print("Phase 3: 困惑度分析與閾值設定")
    print("="*60)
    
    # 載入資料
    with open('training_questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 分類好壞題目
    good_questions = [item['question'] for item in data if item['quality_score'] >= 7]
    bad_questions = [item['question'] for item in data if item['quality_score'] < 5]
    
    print(f"高品質題目：{len(good_questions)} 個")
    print(f"低品質題目：{len(bad_questions)} 個")
    
    # 初始化分析器
    analyzer = PerplexityAnalyzer(model_name="gpt2")
    
    # 視覺化分布
    analyzer.visualize_perplexity_distribution(good_questions, bad_questions)
    
    # 找最佳閾值
    all_questions = good_questions + bad_questions
    labels = [True] * len(good_questions) + [False] * len(bad_questions)
    
    threshold_finder = PerplexityThresholdFinder(analyzer)
    best_threshold = threshold_finder.find_optimal_threshold(all_questions, labels)
    
    print(f"\n建議困惑度閾值：{best_threshold['threshold']:.2f}")
    print(f"F1 分數：{best_threshold['f1']:.3f}")
    
    return best_threshold['threshold']


# ============================================================================
# Phase 4: 整合系統並生成題目
# ============================================================================

def generate_high_quality_questions(
    topic: str,
    difficulty: str = "中等",
    count: int = 20,
    api_key: str = None
):
    """生成高品質題目（完整流程）"""
    print("\n" + "="*60)
    print(f"Phase 4: 生成高品質題目 - {topic}")
    print("="*60)
    
    # 載入已訓練的模型
    lasso_selector = load_model('lasso_quality_model.pkl')
    
    # 初始化系統
    system = IntelligentQuestionSystem(
        api_provider="openai",
        api_key=api_key,
        lasso_alpha=lasso_selector.alpha,
        perplexity_threshold=50.0,  # 使用 Phase 3 分析的結果
        temperature=0.2  # 低溫生成
    )
    
    # 載入訓練資料（用於參考）
    with open('training_questions.json', 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    training_questions = [item['question'] for item in training_data]
    training_scores = [item['quality_score'] for item in training_data]
    
    # 訓練系統內部模型
    system.train_quality_model(training_questions, training_scores)
    
    # 生成題目
    questions = system.generate_and_filter(
        topic=topic,
        difficulty=difficulty,
        target_count=count,
        candidate_multiplier=5  # 生成 5 倍候選進行篩選
    )
    
    # 匯出結果
    output_file = f"generated_questions_{topic}_{difficulty}.json"
    system.export_results(questions, output_file)
    
    # 顯示結果摘要
    print_question_summary(questions)
    
    return questions


def print_question_summary(questions: list):
    """列印題目摘要"""
    print("\n" + "="*60)
    print("生成結果摘要")
    print("="*60)
    
    # 品質分數統計
    quality_scores = [q['quality_score'] for q in questions]
    perplexities = [q['perplexity'] for q in questions]
    
    print(f"\n品質分數：")
    print(f"  平均：{sum(quality_scores)/len(quality_scores):.2f}")
    print(f"  最高：{max(quality_scores):.2f}")
    print(f"  最低：{min(quality_scores):.2f}")
    
    print(f"\n困惑度：")
    print(f"  平均：{sum(perplexities)/len(perplexities):.2f}")
    print(f"  最低：{min(perplexities):.2f}")
    print(f"  最高：{max(perplexities):.2f}")
    
    # 顯示前 3 個題目
    print(f"\n品質最高的 3 個題目：")
    sorted_questions = sorted(questions, key=lambda x: x['quality_score'], reverse=True)
    for i, q in enumerate(sorted_questions[:3], 1):
        print(f"\n{i}. (品質分數: {q['quality_score']:.2f}, 困惑度: {q['perplexity']:.2f})")
        print(f"   {q['question'][:80]}...")


# ============================================================================
# Phase 5: 批次生成多科目題目
# ============================================================================

def batch_generate_questions(
    topics_config: list,
    api_key: str = None
):
    """批次生成多個科目的題目"""
    print("\n" + "="*60)
    print("Phase 5: 批次生成多科目題目")
    print("="*60)
    
    all_results = {}
    
    for config in topics_config:
        topic = config['topic']
        difficulty = config.get('difficulty', '中等')
        count = config.get('count', 20)
        
        print(f"\n處理：{topic} ({difficulty}) - {count} 題")
        
        questions = generate_high_quality_questions(
            topic=topic,
            difficulty=difficulty,
            count=count,
            api_key=api_key
        )
        
        all_results[topic] = questions
    
    # 統計總結
    print("\n" + "="*60)
    print("批次生成總結")
    print("="*60)
    for topic, questions in all_results.items():
        avg_quality = sum(q['quality_score'] for q in questions) / len(questions)
        print(f"{topic}: {len(questions)} 題, 平均品質 {avg_quality:.2f}")
    
    return all_results


# ============================================================================
# 主流程：從零到完整系統
# ============================================================================

def main():
    """完整流程範例"""
    
    # ========== 設定 ==========
    API_KEY = "your-openai-api-key-here"
    
    # ========== Phase 1: 準備資料 ==========
    print("\n開始 Phase 1: 準備訓練資料...")
    prepare_initial_training_data()
    
    # ========== Phase 2: 訓練模型 ==========
    print("\n開始 Phase 2: 訓練品質模型...")
    train_quality_model()
    
    # ========== Phase 3: 困惑度分析 ==========
    print("\n開始 Phase 3: 困惑度分析...")
    optimal_threshold = analyze_and_set_perplexity_threshold()
    
    # ========== Phase 4: 生成單一科目 ==========
    print("\n開始 Phase 4: 生成題目...")
    questions = generate_high_quality_questions(
        topic="憲法基本權利",
        difficulty="中等",
        count=20,
        api_key=API_KEY
    )
    
    # ========== Phase 5: 批次生成（可選）==========
    print("\n開始 Phase 5: 批次生成...")
    topics = [
        {"topic": "憲法基本權利", "difficulty": "中等", "count": 20},
        {"topic": "行政法總論", "difficulty": "困難", "count": 15},
        {"topic": "刑法總則", "difficulty": "簡單", "count": 25},
    ]
    batch_generate_questions(topics, api_key=API_KEY)
    
    print("\n" + "="*60)
    print("所有流程完成！")
    print("="*60)


# ============================================================================
# 快速開始：已有訓練好的模型
# ============================================================================

def quick_start(api_key: str):
    """
    快速開始（假設已經訓練好模型）
    """
    questions = generate_high_quality_questions(
        topic="憲法基本權利",
        difficulty="中等",
        count=20,
        api_key=api_key
    )
    
    return questions


if __name__ == "__main__":
    # 選擇執行模式
    
    # 模式 1: 完整流程（首次使用）
    # main()
    
    # 模式 2: 快速生成（已訓練模型）
    API_KEY = "your-api-key-here"
    questions = quick_start(API_KEY)
    
    print(f"\n生成了 {len(questions)} 個高品質題目！")
