# 📁 專案結構說明

## 完整目錄樹

```
intelligent-question-system-v2/
│
├── 📂 .github/                        # GitHub 配置
│   └── workflows/
│       └── ci.yml                     # 自動化測試流程
│
├── 📂 src/                            # 核心程式碼
│   ├── __init__.py                    # Package 初始化
│   ├── intelligent_question_system.py # 🔥 主系統（21 KB）
│   ├── train_lasso_model.py           # 🔥 Lasso 訓練（11 KB）
│   └── perplexity_analysis.py         # 🔥 困惑度分析（13 KB）
│
├── 📂 examples/                       # 使用範例
│   └── complete_example.py            # 🔥 完整流程範例（11 KB）
│
├── 📂 data/                           # 資料目錄
│   └── training_questions_template.json # 訓練資料範本（20 題）
│
├── 📂 docs/                           # 文件目錄
│   └── QUICK_START.md                 # 快速開始指南
│
├── 📂 tests/                          # 測試目錄
│   └── test_system.py                 # 測試範本（待開發）
│
├── 📄 .gitignore                      # Git 忽略檔案清單
├── 📄 CONTRIBUTING.md                 # 貢獻指南
├── 📄 LICENSE                         # MIT 授權條款
├── 📄 README.md                       # 📖 完整說明文件（11 KB）
├── 📄 README_GITHUB.md                # GitHub 首頁用（簡化版）
├── 📄 config_template.py              # 配置檔範本
├── 📄 requirements.txt                # Python 套件列表
└── 📄 setup.py                        # 安裝腳本
```

---

## 核心檔案說明

### 🔥 核心程式碼（src/）

#### `intelligent_question_system.py`
**主系統整合檔案**，包含：
- `IntelligentQuestionSystem` 類別：完整的題目生成流程
- `LowTempQuestionGenerator`：低溫題目生成器
- `QuestionFeatureExtractor`：特徵提取器（15 個特徵）
- `LassoQuestionSelector`：Lasso 品質選擇器

**使用場景：**
```python
from src.intelligent_question_system import IntelligentQuestionSystem
system = IntelligentQuestionSystem(temperature=0.2)
questions = system.generate_and_filter(topic="憲法", count=20)
```

#### `train_lasso_model.py`
**Lasso 模型訓練**，包含：
- `QuestionQualityTrainer`：品質模型訓練器
- 交叉驗證
- 特徵重要性分析
- 視覺化工具

**使用場景：**
```python
from src.train_lasso_model import QuestionQualityTrainer
trainer = QuestionQualityTrainer()
lasso_selector, results = trainer.train_with_cross_validation(features, scores)
```

#### `perplexity_analysis.py`
**困惑度計算與分析**，包含：
- `PerplexityCalculator`：困惑度計算器
- `PerplexityAnalyzer`：詳細分析工具
- `PerplexityThresholdFinder`：最佳閾值尋找器

**使用場景：**
```python
from src.perplexity_analysis import PerplexityAnalyzer
analyzer = PerplexityAnalyzer()
result = analyzer.calculate_perplexity(question)
```

---

### 📖 文件檔案

#### `README.md` (11 KB)
**完整說明文件**，包含：
- 系統介紹
- 安裝指南
- 使用範例
- API 文件
- 參數調整指南
- FAQ

#### `README_GITHUB.md`
**GitHub 首頁用**（簡化版），包含：
- 快速介紹
- 核心特色
- 快速開始
- 專案結構
- Badges

#### `docs/QUICK_START.md`
**快速開始指南**，包含：
- 3 分鐘快速開始
- 常見調整
- 問題排查

---

### 🔧 配置檔案

#### `config_template.py`
**配置範本**，包含：
- API Keys 設定
- 預設參數
- 路徑配置

**使用方式：**
```bash
cp config_template.py config.py
# 編輯 config.py 填入你的 API Keys
```

#### `requirements.txt`
**Python 套件相依性**：
- numpy, pandas, scikit-learn
- torch, transformers
- openai, anthropic
- matplotlib, seaborn

---

### 📊 資料檔案

#### `data/training_questions_template.json`
**訓練資料範本**（20 題範例），格式：
```json
[
  {
    "question": "題目內容...",
    "quality_score": 8.5,
    "comment": "評分說明"
  }
]
```

---

### 🧪 測試檔案

#### `tests/test_system.py`
**測試範本**（待開發），包含：
- 單元測試框架
- 測試範例
- pytest 配置

---

### ⚙️ GitHub 配置

#### `.github/workflows/ci.yml`
**自動化測試流程**：
- Python 3.8-3.11 測試
- Code linting (flake8)
- Code formatting (black)
- 測試覆蓋率

#### `.gitignore`
**Git 忽略清單**：
- Python cache 檔案
- 虛擬環境
- API Keys
- 模型檔案
- 輸出檔案

---

## 檔案大小統計

| 類型 | 檔案數 | 總大小 |
|------|--------|--------|
| 核心程式碼 | 4 | ~56 KB |
| 文件 | 5 | ~25 KB |
| 配置 | 4 | ~3 KB |
| 資料範本 | 1 | ~5 KB |
| 總計 | 16 | ~89 KB |

---

## 工作流程圖

```
📥 輸入主題
    ↓
📂 src/intelligent_question_system.py
    ├─ LowTempQuestionGenerator (生成)
    ├─ PerplexityCalculator (篩選)
    └─ LassoQuestionSelector (排序)
    ↓
📤 輸出高品質題目
```

---

## 擴展指南

### 新增功能
1. 在 `src/` 創建新的 Python 檔案
2. 在 `src/__init__.py` 添加 import
3. 在 `examples/` 創建使用範例
4. 在 `tests/` 添加測試

### 新增文件
1. 在 `docs/` 創建 Markdown 檔案
2. 在 `README.md` 添加連結

### 新增資料
1. 在 `data/` 放置資料檔案
2. 更新 `.gitignore`（如需忽略）
3. 在 `README.md` 說明資料格式

---

## 依賴關係圖

```
intelligent_question_system.py
    ├─ train_lasso_model.py
    │   └─ QuestionFeatureExtractor
    └─ perplexity_analysis.py
        └─ PerplexityCalculator

complete_example.py
    └─ intelligent_question_system.py
```
