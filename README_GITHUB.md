# 🎓 智慧命題 2.0 系統

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)]()

使用 **Lasso Regression + Perplexity + Low Temperature** 的科學化題目生成與品質控制系統。

---

## ✨ 核心特色

- 🎯 **科學化品質控制**：使用困惑度量化題目品質
- 🔬 **Lasso 特徵選擇**：自動識別 5-8 個最重要特徵
- ❄️ **低溫生成**：Temperature=0.2，確保輸出穩定
- 📊 **可解釋性**：知道每個特徵對品質的貢獻
- 💰 **成本優化**：相比多模型投票，降低 60% API 成本

---

## 🚀 快速開始

### 安裝

```bash
git clone https://github.com/ritalinyutzu/intelligent-question-system.git
cd intelligent-question-system
pip install -r requirements.txt
```

### 使用範例

```python
from src.intelligent_question_system import IntelligentQuestionSystem

# 初始化系統
system = IntelligentQuestionSystem(
    api_provider="openai",
    api_key="your-api-key",
    temperature=0.2,
    perplexity_threshold=50.0
)

# 訓練模型
system.train_quality_model(training_questions, quality_scores)

# 生成高品質題目
questions = system.generate_and_filter(
    topic="憲法基本權利",
    difficulty="中等",
    target_count=20
)
```

---

## 📊 系統架構

```
低溫生成（T=0.2）
    ↓
生成候選題目 ×5
    ↓
困惑度篩選（PPL < 50）
    ↓
Lasso 品質評分
    ↓
返回 Top N 題目
```

---

## 📁 專案結構

```
intelligent-question-system-v2/
├── src/                                    # 核心程式碼
│   ├── intelligent_question_system.py     # 主系統
│   ├── train_lasso_model.py               # Lasso 訓練
│   └── perplexity_analysis.py             # 困惑度分析
├── examples/                               # 使用範例
│   └── complete_example.py                # 完整流程範例
├── data/                                   # 資料
│   └── training_questions_template.json   # 訓練資料範本
├── docs/                                   # 文件
│   └── QUICK_START.md                     # 快速開始指南
├── tests/                                  # 測試（待開發）
├── README.md                               # 完整說明
├── requirements.txt                        # 相依套件
├── setup.py                                # 安裝腳本
└── LICENSE                                 # MIT 授權
```

---

## 🎯 核心技術

| 技術 | 用途 | 推薦值 |
|------|------|--------|
| **Low Temperature** | 生成穩定性 | 0.2 |
| **Perplexity** | 品質篩選 | < 50 |
| **Lasso Alpha** | 特徵選擇 | 0.1 |

---

## 📖 完整文件

- [完整說明文件](README.md)
- [快速開始指南](docs/QUICK_START.md)
- [使用範例](examples/complete_example.py)

---

## 🆚 對比傳統方法

| 項目 | 傳統（多模型投票） | 智慧命題 2.0 |
|------|-------------------|-------------|
| API 成本 | 高（3 倍） | ✅ 降低 60% |
| 生成速度 | 慢 | ✅ 快 |
| 品質控制 | 主觀 | ✅ 客觀量化 |
| 可解釋性 | 黑箱 | ✅ 透明 |
| 可調整性 | 低 | ✅ 高 |

---

## 💡 使用場景

- 📚 國家考試（高考、普考、特考）
- 🎓 證照考試（金融、IT、醫療）
- 🏫 學校測驗（大學、高中、國中）
- 📖 線上學習平台
- 🧠 企業內訓測驗

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

---

## 📧 聯絡方式

- **作者**: Rita Lin
- **Email**: msmile09@hotmail.com
- **GitHub**: [@ritalinyutzu](https://github.com/ritalinyutzu)

---

## 📜 授權

本專案採用 [MIT License](LICENSE)
