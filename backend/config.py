# ============================================
# ZeroCode — config.py
# Project config and environment variables.
# ============================================

from dotenv import load_dotenv
import os

load_dotenv()

# Keep tracing in the local Jaeger collector rather than CrewAI's hosted telemetry.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

PROJECT_NAME = "ZeroCode"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME: str = "gemini-3.6-flash"
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "outputs"))
MESSAGING_MODE = os.getenv("MESSAGING_MODE", "blackboard")
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "sequential")
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".csv"]
