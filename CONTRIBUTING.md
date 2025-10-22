# 貢獻指南

感謝你對智慧命題 2.0 系統的興趣！

## 如何貢獻

### 回報 Bug

請在 GitHub Issues 中建立新的 issue，並包含：
- 問題描述
- 重現步驟
- 預期行為
- 實際行為
- 環境資訊（Python 版本、作業系統等）

### 提交功能請求

在 GitHub Issues 中描述：
- 功能需求
- 使用場景
- 預期效果

### 提交程式碼

1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. Push 到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

### 程式碼風格

- 遵循 PEP 8
- 使用有意義的變數名稱
- 添加適當的註解和文件字串
- 確保所有測試通過

### 測試

```bash
pytest tests/
```

## 開發環境設置

```bash
# Clone 專案
git clone https://github.com/ritalinyutzu/intelligent-question-system.git
cd intelligent-question-system

# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝相依套件
pip install -r requirements.txt
pip install -e .  # 開發模式安裝

# 執行測試
pytest tests/
```

## 聯絡方式

有任何問題？歡迎聯絡：
- Email: msmile09@hotmail.com
- GitHub: @ritalinyutzu

感謝你的貢獻！🎉
