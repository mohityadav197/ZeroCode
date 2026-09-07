# ============================================
# ZeroCode — config.py
# Project config and environment variables.
# ============================================

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_NAME = "ZeroCode"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "openai/gpt-oss-20b"
UPLOAD_FOLDER = "outputs"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".csv"]
