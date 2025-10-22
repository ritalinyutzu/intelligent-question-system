# 配置檔案範本
# 複製此檔案為 config.py 並填入你的 API Keys

# API 配置
OPENAI_API_KEY = "your-openai-api-key-here"
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
GOOGLE_API_KEY = "your-google-api-key-here"

# 模型參數
DEFAULT_TEMPERATURE = 0.2
DEFAULT_PERPLEXITY_THRESHOLD = 50.0
DEFAULT_LASSO_ALPHA = 0.1

# 生成參數
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_TARGET_COUNT = 20

# 路徑配置
TRAINING_DATA_PATH = "data/training_questions.json"
MODEL_SAVE_PATH = "models/lasso_quality_model.pkl"
OUTPUT_DIR = "outputs/"
