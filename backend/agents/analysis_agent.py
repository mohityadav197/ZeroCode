# ============================================
# ZeroCode — analysis_agent.py
# Analysis Agent — profiles the dataset, detects
# column types, suggests target column, detects
# problem type, generates EDA charts and LLM insights.
# ============================================

import os

import kaleido
import numpy as np
import pandas as pd
import plotly.express as px

from config import UPLOAD_FOLDER
from llm_provider import LLMProvider

MAX_BAR_CHART_CATEGORIES = 20

CHARTS_FOLDER = os.path.join(UPLOAD_FOLDER, "charts")

TARGET_KEYWORDS = [
    "target", "label", "output", "class", "result", "price", "salary",
    "survived", "churn", "score", "revenue", "sales", "grade", "status",
    "approved", "converted",
]


class AnalysisAgent:
    def __init__(self):
        self.llm = LLMProvider()

    # ------------------------------------------------------------------
    def profile_data(self, df: pd.DataFrame) -> dict:
        rows = len(df)
        columns = len(df.columns)

        column_types = {}
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                column_types[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                column_types[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                column_types[col] = "numeric"
            else:
                column_types[col] = "categorical"

        missing = {
            col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns
        }
        unique_counts = {col: int(df[col].nunique()) for col in df.columns}

        constant_columns = [col for col, cnt in unique_counts.items() if cnt == 1]
        id_like_columns = [
            col for col, cnt in unique_counts.items() if rows > 0 and cnt == rows
        ]

        quality_score = 100
        for col, pct in missing.items():
            if pct > 5:
                quality_score -= 5
            if pct > 50:
                quality_score -= 10
        if constant_columns:
            quality_score -= 5
        if id_like_columns:
            quality_score -= 5
        quality_score = max(0, quality_score)

        return {
            "rows": rows,
            "columns": columns,
            "column_types": column_types,
            "missing": missing,
            "unique_counts": unique_counts,
            "constant_columns": constant_columns,
            "id_like_columns": id_like_columns,
            "quality_score": quality_score,
        }

    # ------------------------------------------------------------------
    def suggest_target(self, df: pd.DataFrame, profile: dict) -> dict:
        columns = list(df.columns)
        unique_counts = profile["unique_counts"]
        constant_columns = set(profile["constant_columns"])
        id_like_columns = set(profile["id_like_columns"])

        scores = {}
        for i, col in enumerate(columns):
            score = 0
            col_lower = col.lower()

            if any(keyword in col_lower for keyword in TARGET_KEYWORDS):
                score += 50

            is_last = i == len(columns) - 1
            uniq = unique_counts[col]

            if is_last and uniq < 20:
                score += 20

            if 2 <= uniq <= 20:
                score += 15

            if col in constant_columns or col in id_like_columns:
                score -= 100

            scores[col] = score

        suggested_target = max(scores, key=scores.get)
        best_score = scores[suggested_target]

        if best_score < 0:
            suggested_target = columns[-1]
            best_score = scores[suggested_target]

        col_lower = suggested_target.lower()
        uniq = unique_counts[suggested_target]
        is_last = columns[-1] == suggested_target

        if any(keyword in col_lower for keyword in TARGET_KEYWORDS):
            reason = f"Column name '{suggested_target}' matches a common target keyword."
        elif is_last and uniq < 20:
            reason = f"'{suggested_target}' is the last column and has only {uniq} unique values, a common target convention."
        elif 2 <= uniq <= 20:
            reason = f"'{suggested_target}' has {uniq} unique values, typical of a classification target."
        else:
            reason = f"'{suggested_target}' was chosen as a fallback with no stronger candidate found."

        if best_score >= 50:
            confidence = "high"
        elif best_score >= 15:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "suggested_target": suggested_target,
            "reason": reason,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    def detect_problem_type(self, df: pd.DataFrame, target_col: str) -> dict:
        target_unique = int(df[target_col].nunique())

        if target_unique == 2:
            problem_type = "binary_classification"
            explanation = f"Target column '{target_col}' has exactly 2 unique values, indicating binary classification."
        elif 3 <= target_unique <= 20:
            problem_type = "multiclass_classification"
            explanation = f"Target column '{target_col}' has {target_unique} unique values, indicating multiclass classification."
        elif pd.api.types.is_numeric_dtype(df[target_col]):
            problem_type = "regression"
            explanation = f"Target column '{target_col}' is numeric with {target_unique} unique values, indicating a regression problem."
        else:
            problem_type = "multiclass_classification"
            explanation = f"Target column '{target_col}' is categorical with {target_unique} unique values."

        return {
            "problem_type": problem_type,
            "target_unique_values": target_unique,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    def generate_charts(self, df: pd.DataFrame, profile: dict) -> dict:
        os.makedirs(CHARTS_FOLDER, exist_ok=True)

        column_types = profile["column_types"]
        unique_counts = profile.get("unique_counts", {})
        id_like_columns = set(profile.get("id_like_columns", []))

        numeric_cols = [c for c, t in column_types.items() if t == "numeric" and c not in id_like_columns]
        categorical_cols = [
            c for c, t in column_types.items()
            if t == "categorical" and unique_counts.get(c, 0) <= MAX_BAR_CHART_CATEGORIES
        ]
        corr_cols = [c for c, t in column_types.items() if t == "numeric"]

        charts = {}

        def save_chart(fig, key: str):
            filename = f"{key}.png"
            filepath = os.path.join(CHARTS_FOLDER, filename)
            try:
                fig.write_image(filepath)
                charts[key] = f"outputs/charts/{filename}"
            except Exception:
                pass

        # Reuse a single Chrome instance for every chart export instead of
        # launching a new browser per fig.write_image() call.
        kaleido.start_sync_server(silence_warnings=True)
        try:
            for col in numeric_cols:
                try:
                    fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                    save_chart(fig, f"histogram_{col}")
                except Exception:
                    pass

                try:
                    fig = px.box(df, y=col, title=f"Boxplot of {col}")
                    save_chart(fig, f"boxplot_{col}")
                except Exception:
                    pass

            for col in categorical_cols:
                try:
                    counts = df[col].value_counts()
                    fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Value counts of {col}")
                    save_chart(fig, f"bar_{col}")
                except Exception:
                    pass

            if len(corr_cols) > 1:
                try:
                    corr = df[corr_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                    save_chart(fig, "correlation_heatmap")
                except Exception:
                    pass
        finally:
            kaleido.stop_sync_server()

        return {
            "charts": charts,
            "chart_count": len(charts),
        }

    # ------------------------------------------------------------------
    def generate_eda_code(self, df: pd.DataFrame, profile: dict) -> str:
        column_types = profile.get("column_types", {})
        numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
        categorical_cols = [c for c, t in column_types.items() if t == "categorical"]

        template = '''# Auto-generated ZeroCode EDA script
# Reproduces the exploratory charts and stats generated by the Analysis Agent

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load your dataset here
DATA_PATH = "your_dataset.csv"
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------------
# Basic info
print("Shape:", df.shape)
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values (%):\\n", (df.isna().mean() * 100).round(2))

# ------------------------------------------------------------------
# Column groups detected during analysis
numeric_cols = __NUMERIC_COLS__
categorical_cols = __CATEGORICAL_COLS__

# ------------------------------------------------------------------
# Histograms and boxplots for numeric columns (outlier detection)
for col in numeric_cols:
    fig = px.histogram(df, x=col, title=f"Histogram of {col}")
    fig.write_image(f"charts/histogram_{col}.png")

    fig = px.box(df, y=col, title=f"Boxplot of {col}")
    fig.write_image(f"charts/boxplot_{col}.png")

# ------------------------------------------------------------------
# Bar charts (value counts) for categorical columns
for col in categorical_cols:
    counts = df[col].value_counts()
    fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Value counts of {col}")
    fig.write_image(f"charts/bar_{col}.png")

# ------------------------------------------------------------------
# Correlation heatmap
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    print("\\nCorrelation matrix:\\n", corr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("charts/correlation_heatmap_seaborn.png")

# ------------------------------------------------------------------
# Key summary statistics
print("\\nSummary statistics:\\n", df.describe(include="all"))
'''

        code = template.replace("__NUMERIC_COLS__", repr(numeric_cols)).replace(
            "__CATEGORICAL_COLS__", repr(categorical_cols)
        )

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, "eda_report.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        return filepath.replace("\\", "/")

    # ------------------------------------------------------------------
    def detect_outliers(self, df: pd.DataFrame, profile: dict) -> dict:
        numeric_cols = [c for c, t in profile["column_types"].items() if t == "numeric"]
        rows = profile["rows"]

        outliers = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                outliers[col] = {"count": 0, "percentage": 0.0}
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (series < lower_bound) | (series > upper_bound)
            count = int(mask.sum())
            percentage = round((count / rows) * 100, 2) if rows > 0 else 0.0

            outliers[col] = {"count": count, "percentage": percentage}

        return {"outliers": outliers}

    # ------------------------------------------------------------------
    def check_target_balance(self, df: pd.DataFrame, target_col: str) -> dict:
        value_counts = df[target_col].value_counts(normalize=True) * 100

        distribution = {
            f"class_{value}": round(float(pct), 2) for value, pct in value_counts.items()
        }

        if len(distribution) == 0:
            return {"distribution": {}, "is_balanced": True, "warning": None}

        max_pct = max(distribution.values())
        min_pct = min(distribution.values())
        is_balanced = (max_pct - min_pct) <= 20

        warning = None
        if not is_balanced:
            sorted_pcts = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            top = sorted_pcts[0]
            bottom = sorted_pcts[-1]
            warning = (
                f"Target is imbalanced ({top[1]}%/{bottom[1]}%) — "
                "consider using class weights"
            )

        return {
            "distribution": distribution,
            "is_balanced": is_balanced,
            "warning": warning,
        }

    # ------------------------------------------------------------------
    def generate_insights(self, df, profile, outliers, target_info, problem_type) -> list:
        shape_str = f"{profile['rows']} rows x {profile['columns']} columns"
        missing_str = ", ".join(
            f"{col}: {pct}%" for col, pct in profile["missing"].items() if pct > 0
        ) or "none"
        outlier_str = ", ".join(
            f"{col}: {info['count']} ({info['percentage']}%)"
            for col, info in outliers.get("outliers", {}).items()
            if info["count"] > 0
        ) or "none"
        balance_str = target_info.get("distribution", "not applicable")

        numeric_cols = [c for c, t in profile["column_types"].items() if t == "numeric"]
        corr_str = "not computed"
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            pairs = []
            for i, col_a in enumerate(numeric_cols):
                for col_b in numeric_cols[i + 1:]:
                    val = corr.loc[col_a, col_b]
                    if abs(val) > 0.5:
                        pairs.append(f"{col_a}~{col_b}: {round(float(val), 2)}")
            corr_str = ", ".join(pairs) if pairs else "no strong correlations"

        prompt = (
            f"Dataset shape: {shape_str}\n"
            f"Problem type: {problem_type.get('problem_type')}\n"
            f"Missing values: {missing_str}\n"
            f"Outliers: {outlier_str}\n"
            f"Target balance: {balance_str}\n"
            f"Strong correlations: {corr_str}\n"
            f"Data quality score: {profile['quality_score']}/100\n\n"
            "Based on this data profile, give 5 to 7 short, plain-English insights "
            "a data scientist would care about before modeling."
        )

        system_prompt = (
            'Respond with JSON in exactly this shape: {"insights": ["...", "..."]}. '
            "Each insight should be one concise sentence."
        )

        result = self.llm.ask_json(prompt, system_prompt=system_prompt)
        insights = result.get("insights") if isinstance(result, dict) else None

        if insights:
            return insights

        fallback = []
        for col, pct in profile["missing"].items():
            if pct > 0:
                fallback.append(f"{col} has {pct}% missing values and needs handling before training.")
                break
        for col, info in outliers.get("outliers", {}).items():
            if info["count"] > 0:
                fallback.append(f"{col} has {info['count']} outliers ({info['percentage']}%) detected via IQR.")
                break
        if target_info.get("warning"):
            fallback.append(target_info["warning"])

        if not fallback:
            fallback.append(f"Dataset quality score is {profile['quality_score']}/100 with no major issues detected.")

        return fallback[:3] if len(fallback) >= 3 else fallback

    # ------------------------------------------------------------------
    def run_full_analysis(self, df: pd.DataFrame, confirmed_target: str = None) -> dict:
        profile = self.profile_data(df)

        if confirmed_target:
            target_suggestion = {
                "suggested_target": confirmed_target,
                "reason": "User confirmed this target column.",
                "confidence": "high",
            }
        else:
            target_suggestion = self.suggest_target(df, profile)

        target_col = target_suggestion["suggested_target"]

        problem_type = self.detect_problem_type(df, target_col)
        charts = self.generate_charts(df, profile)
        eda_code_path = self.generate_eda_code(df, profile)
        outliers = self.detect_outliers(df, profile)

        if problem_type["problem_type"] in ("binary_classification", "multiclass_classification"):
            target_balance = self.check_target_balance(df, target_col)
        else:
            target_balance = {"distribution": {}, "is_balanced": True, "warning": None}

        insights = self.generate_insights(df, profile, outliers, target_balance, problem_type)

        return {
            "profile": profile,
            "target_suggestion": target_suggestion,
            "problem_type": problem_type,
            "charts": charts,
            "eda_code_path": eda_code_path,
            "outliers": outliers,
            "target_balance": target_balance,
            "insights": insights,
            "status": "completed",
        }
