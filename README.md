# 智慧命題 2.0 系統

使用 **Lasso Regression + Perplexity + Low Temperature** 的科學化題目生成與品質控制系統。

---

## 🎯 核心技術

### 1. **Low Temperature Generation (0.2)**
- 生成更穩定、更貼近標準答案的題目
- 減少模型的隨機性和創意性
- 確保輸出符合考試規範

### 2. **Perplexity (困惑度)**
- 量化題目的"標準程度"
- 困惑度 < 50 = 高品質題目
- 自動過濾不符合語言模型訓練分布的題目

### 3. **Lasso Regression**
- 自動特徵選擇（懲罰無效特徵）
- 從 15+ 個特徵中選出最重要的 5-8 個
- 建立可解釋的品質評估模型

---

## 📦 安裝

```bash
# 安裝相依套件
pip install -r requirements.txt

# 如果要使用 CUDA 加速
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 快速開始

### Step 1: 準備訓練資料

創建 `training_questions.json`:

```json
[
  {
    "question": "下列何者為憲法保障之基本權利？(A)生存權 (B)工作權 (C)財產權 (D)以上皆是",
    "quality_score": 9.0
  },
  {
    "question": "憲法權利有哪些？A生存B工作C財產D全部",
    "quality_score": 4.0
  }
]
```

**建議至少 50-100 題**，品質分數 1-10 分。

---

### Step 2: 訓練品質模型

```python
from train_lasso_model import QuestionQualityTrainer, save_model

# 初始化
trainer = QuestionQualityTrainer()

# 載入資料
questions, scores = load_training_data('training_questions.json')

# 準備特徵
features_df, scores = trainer.prepare_training_data(questions, scores)

# 訓練模型
lasso_selector, cv_results = trainer.train_with_cross_validation(features_df, scores)

# 儲存模型
save_model(lasso_selector, 'lasso_quality_model.pkl')
```

**輸出：**
```
特徵選擇結果：
原始特徵數：15
選中特徵數：7
被選中的特徵：['perplexity', 'length', 'has_options', 'uniqueness', ...]

特徵重要性：
  perplexity                    :  -0.3245
  uniqueness                    :   0.2891
  has_options                   :   0.2156
  ...

✓ 模型已儲存：lasso_quality_model.pkl
```

---

### Step 3: 分析困惑度

```python
from perplexity_analysis import PerplexityAnalyzer, PerplexityThresholdFinder

# 初始化
analyzer = PerplexityAnalyzer(model_name="gpt2")

# 分析單題
result = analyzer.calculate_perplexity(question, verbose=True)

# 找最佳閾值
threshold_finder = PerplexityThresholdFinder(analyzer)
best_threshold = threshold_finder.find_optimal_threshold(questions, labels)

print(f"建議困惑度閾值：{best_threshold['threshold']:.2f}")
```

**輸出：**
```
建議困惑度閾值：48.50
F1 分數：0.872
```

---

### Step 4: 生成高品質題目

```python
from intelligent_question_system import IntelligentQuestionSystem

# 初始化系統
system = IntelligentQuestionSystem(
    api_provider="openai",
    api_key="your-api-key",
    lasso_alpha=0.1,
    perplexity_threshold=50.0,
    temperature=0.2  # 低溫！
)

# 訓練系統
system.train_quality_model(training_questions, training_scores)

# 生成題目
questions = system.generate_and_filter(
    topic="憲法基本權利",
    difficulty="中等",
    target_count=20,
    candidate_multiplier=5  # 生成 100 個候選，篩選出 20 個
)

# 匯出
system.export_results(questions, "output.json")
```

**輸出：**
```
============================================================
開始生成題目：憲法基本權利 / 難度：中等
目標題目數：20
============================================================

Step 1: 生成 100 個候選題目（低溫 0.2）...
✓ 生成完成：100 個候選題目

Step 2: 困惑度篩選（閾值 < 50.0）...
✓ 通過困惑度篩選：73/100 個
  平均困惑度：42.35
  困惑度範圍：18.23 ~ 89.45

Step 3: Lasso 品質評估與排序...
✓ 品質評分完成
  平均品質分數：7.82
  分數範圍：5.21 ~ 9.67

Step 4: 選擇品質最高的 20 個題目

============================================================
完成！最終產出 20 個高品質題目
============================================================
```

---

## 📊 系統架構

```
輸入：主題 + 難度
    ↓
┌─────────────────────────────────┐
│  Step 1: 低溫生成（Temperature=0.2） │
│  生成 N×5 個候選題目                  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Step 2: 困惑度篩選（PPL < 50）      │
│  過濾語言模型不熟悉的題目              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Step 3: Lasso 特徵選擇與品質評估     │
│  使用 7 個關鍵特徵預測品質分數          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Step 4: 排序並返回 Top N            │
│  輸出：高品質題目 + 品質分數 + 困惑度    │
└─────────────────────────────────┘
```

---

## 📈 特徵工程

系統提取 **15 個特徵**，Lasso 自動選出最重要的 **5-8 個**：

### 基礎特徵
- `length`: 題目長度
- `word_count`: 字詞數量
- `avg_word_length`: 平均字長
- `char_count`: 字元數量
- `punctuation_count`: 標點符號數量

### 語義特徵
- `exam_keyword_density`: 考試關鍵詞密度
- `professional_term_count`: 專業術語數量

### 結構特徵
- `has_options`: 是否有選項
- `option_count`: 選項數量
- `has_clear_structure`: 是否結構清晰
- `has_question_mark`: 是否有問號

### 品質特徵
- `perplexity`: 困惑度（最重要！）
- `uniqueness`: 獨特性（與現有題庫的差異度）

### 答案特徵（可選）
- `answer_length`: 答案長度
- `has_explanation`: 是否有詳細解析

---

## 🔧 參數調整指南

### Temperature（溫度）

| 溫度 | 效果 | 適用場景 |
|------|------|---------|
| 0.1-0.3 | 非常穩定，貼近標準 | 國考、證照考 |
| 0.4-0.6 | 平衡創意與穩定 | 學校考試 |
| 0.7-1.0 | 有創意但可能偏離 | 腦力激盪 |

**建議：0.2**

---

### Perplexity Threshold（困惑度閾值）

| 閾值 | 篩選效果 | 通過率 |
|------|---------|--------|
| < 30 | 極嚴格，只保留最標準的題目 | ~20% |
| < 50 | 嚴格，保留高品質題目 | ~40% |
| < 80 | 寬鬆，允許較多變化 | ~70% |
| < 100 | 很寬鬆 | ~90% |

**建議：50**

---

### Lasso Alpha（懲罰係數）

| Alpha | 特徵選擇 | 模型複雜度 |
|-------|---------|-----------|
| 0.01 | 保留大部分特徵（12-15個） | 高 |
| 0.05 | 適中（8-10個） | 中 |
| 0.1 | 嚴格（5-8個） | 低 |
| 0.5+ | 非常嚴格（2-4個） | 很低 |

**建議：0.1**（透過交叉驗證自動選擇）

---

## 📁 檔案結構

```
intelligent-question-system/
├── intelligent_question_system.py    # 主系統
├── train_lasso_model.py              # Lasso 訓練
├── perplexity_analysis.py            # 困惑度分析
├── complete_example.py               # 完整範例
├── requirements.txt                  # 相依套件
├── README.md                         # 說明文件
│
├── training_questions.json           # 訓練資料
├── lasso_quality_model.pkl           # 訓練好的模型
│
└── outputs/                          # 產出目錄
    ├── generated_questions_*.json    # 生成的題目
    ├── feature_importance.png        # 特徵重要性圖
    ├── prediction_analysis.png       # 預測分析圖
    └── perplexity_comparison.png     # 困惑度比較圖
```

---

## 🆚 vs 舊版（ChatGPT + Claude + Gemini 投票）

| 項目 | 舊版（3 模型投票） | 新版（Lasso + PPL + Low Temp） |
|------|-------------------|-------------------------------|
| **API 成本** | 高（3 倍） | 中（1 倍 + 篩選） |
| **生成速度** | 慢（需等待 3 次） | 快（單次生成 + 快速篩選） |
| **品質控制** | 主觀（投票） | 客觀（數值化指標） |
| **可解釋性** | 低（黑箱） | 高（知道每個特徵的貢獻） |
| **可調整性** | 低 | 高（可調整閾值和權重） |
| **一致性** | 中等 | 高（相同輸入 → 相同輸出） |

---

## 💡 最佳實踐

### 1. 訓練資料準備
- **最少 50 題**，建議 100-200 題
- 涵蓋不同難度和主題
- 人工標註品質分數（1-10）
- 包含好題目和壞題目（用於學習差異）

### 2. 定期更新模型
- 每月新增新題目到訓練集
- 重新訓練 Lasso 模型
- 調整困惑度閾值

### 3. 批次生成
- 一次生成 5 倍候選數（例如要 20 題，生成 100 個候選）
- 使用多執行緒加速

### 4. 人工複審
- 系統產出後，仍建議人工複審前 20% 的題目
- 將複審結果加入訓練集

---

## 🐛 常見問題

### Q1: 困惑度太高怎麼辦？
**A:** 
1. 檢查 temperature 是否過高（建議 ≤ 0.3）
2. 調高 perplexity_threshold（例如從 50 調到 80）
3. 改用更大的語言模型（gpt2 → gpt2-medium）

### Q2: Lasso 選出的特徵太少？
**A:**
1. 降低 alpha（例如從 0.1 降到 0.05）
2. 檢查訓練資料是否有足夠變異性
3. 新增更多特徵

### Q3: 生成的題目重複性高？
**A:**
1. 提高 `uniqueness` 特徵的權重
2. 在 prompt 中要求多樣性
3. 增加候選倍數（從 5 倍提高到 10 倍）

### Q4: 品質分數預測不準？
**A:**
1. 增加訓練資料數量
2. 檢查人工標註是否一致
3. 使用交叉驗證找最佳 alpha

---

## 📧 聯絡方式

有問題或建議？歡迎聯絡：
- Email: msmile09@hotmail.com
- GitHub: @ritalinyutzu

---

## 📜 授權

MIT License
