import os

class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    MAX_QUESTION_LENGTH = 500
    MODEL_NAME = "mistral-small-latest"
    TEMPERATURE = 0.4
    MAX_TOKENS = 500 