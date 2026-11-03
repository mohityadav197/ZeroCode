import json

import pandas as pd

from config import MAX_FILE_SIZE_MB
from runtime_context import safe_output_path


def parse_csv(filepath: str) -> dict:
    """Validate a CSV in outputs and return its schema, preview and statistics."""
    from guardrails import InputGuard
    path = safe_output_path(filepath)
    if path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError("A readable .csv file is required")
    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError("CSV exceeds the upload size limit")
    frame = pd.read_csv(path)
    validation = InputGuard().validate_csv_schema(frame)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    # JSON serialization normalizes NaN, numpy scalars and timestamp previews.
    return {
        "filepath": str(path),
        "schema": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "preview": json.loads(frame.head(5).to_json(orient="records")),
        "stats": {"rows": len(frame), "columns": len(frame.columns),
                  "missing": frame.isna().sum().to_dict(),
                  "numeric": json.loads(frame.describe().to_json())},
    }
