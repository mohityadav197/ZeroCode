# ============================================
# ZeroCode — file_handler.py
# Handles file saving and loading.
# ============================================

import os

from config import UPLOAD_FOLDER, MAX_FILE_SIZE_MB


def save_upload(file) -> str:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    raise NotImplementedError
