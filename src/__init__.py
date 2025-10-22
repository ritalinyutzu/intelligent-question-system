"""
Intelligent Question Generation System v2.0
使用 Lasso Regression + Perplexity + Low Temperature
"""

__version__ = "2.0.0"
__author__ = "Rita Lin"
__email__ = "msmile09@hotmail.com"

from .intelligent_question_system import IntelligentQuestionSystem
from .train_lasso_model import QuestionQualityTrainer
from .perplexity_analysis import PerplexityAnalyzer

__all__ = [
    'IntelligentQuestionSystem',
    'QuestionQualityTrainer',
    'PerplexityAnalyzer',
]
