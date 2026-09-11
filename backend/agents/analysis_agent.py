# ============================================
# ZeroCode — analysis_agent.py
# Analysis Agent — profiles the dataset, detects
# column types, suggests target column, detects
# problem type, generates EDA charts and LLM insights.
# ============================================

import os
import re

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

NUMERIC_MISSING_OPTIONS = ["fill_median", "fill_mean", "fill_zero", "drop"]
CATEGORICAL_MISSING_OPTIONS = ["drop", "fill_mode", "fill_unknown"]
OUTLIER_OPTIONS_DEFAULT = ["keep", "cap_iqr"]
OUTLIER_OPTIONS_HIGH = ["cap_iqr", "log_transform", "keep"]
SCALING_OPTIONS = ["standard_scale", "minmax_scale", "none"]

# Kept in sync with ml_agent.py's SCALE_SENSITIVE_MODELS / SCALE_INVARIANT_MODELS /
# OUTLIER_SENSITIVE_MODELS — these drive the *suggestion* shown here, ml_agent's
# copies drive what actually happens when the user doesn't override.
SCALE_SENSITIVE_MODELS = {"SVM", "KNN", "Logistic Regression"}
SCALE_INVARIANT_MODELS = {"Random Forest", "XGBoost"}
OUTLIER_SENSITIVE_MODELS = {"SVM", "KNN"}

DOMAIN_KEYWORDS = {
    "healthcare": [
        "patient", "diagnosis", "disease", "symptom",
        "treatment", "hospital", "doctor", "medical",
        "blood", "pressure", "glucose", "bmi", "age",
        "cholesterol", "heart", "cancer", "diabetes",
        "mortality", "survival", "clinical", "drug",
        "dosage", "weight", "height", "pulse",
    ],
    "finance": [
        "price", "revenue", "profit", "loss", "sales",
        "income", "expense", "cost", "loan", "credit",
        "debt", "interest", "rate", "stock", "market",
        "investment", "return", "tax", "salary", "wage",
        "transaction", "balance", "account", "fraud",
        "payment", "budget", "forecast", "gdp", "bank",
    ],
    "hr": [
        "employee", "department", "salary", "hire",
        "attrition", "performance", "rating", "manager",
        "tenure", "promotion", "job", "role", "gender",
        "education", "experience", "training", "leave",
        "satisfaction", "engagement", "headcount", "team",
    ],
    "education": [
        "student", "grade", "score", "marks", "exam",
        "pass", "fail", "gpa", "course", "subject",
        "teacher", "school", "university", "attendance",
        "assignment", "test", "result", "class", "study",
    ],
    "ecommerce": [
        "product", "order", "purchase", "cart", "item",
        "category", "brand", "review", "rating", "customer",
        "churn", "retention", "discount", "quantity", "ship",
        "delivery", "return", "refund", "seller", "listing",
    ],
    "marketing": [
        "campaign", "click", "impression", "conversion",
        "ctr", "cpc", "roi", "channel", "ad", "email",
        "open_rate", "bounce", "subscriber", "lead",
        "funnel", "acquisition", "engagement", "reach",
        "social", "content", "seo", "traffic", "session",
    ],
}

DOMAIN_CONTEXT = {
    "healthcare": {
        "display_name": "Healthcare / Medical",
        "icon": "🏥",
        "description": "This appears to be a medical or clinical dataset",
        "insight_language": "clinical",
        "recommended_targets": ["diagnosis", "disease", "mortality", "survival", "outcome"],
        "domain_tips": [
            "Consider class imbalance — disease datasets often have more negative cases",
            "Feature importance matters here — clinicians need to understand model decisions",
            "SHAP analysis is especially valuable for medical predictions",
        ],
    },
    "finance": {
        "display_name": "Finance / Banking",
        "icon": "💰",
        "description": "This appears to be a financial or banking dataset",
        "insight_language": "financial",
        "recommended_targets": ["fraud", "default", "approved", "churn", "risk"],
        "domain_tips": [
            "Fraud detection datasets are highly imbalanced — consider using SMOTE or class weights",
            "Log transform skewed financial features like income and transaction amounts",
            "ROC-AUC is more important than accuracy for financial risk models",
        ],
    },
    "hr": {
        "display_name": "HR / Human Resources",
        "icon": "👥",
        "description": "This appears to be a human resources or workforce dataset",
        "insight_language": "business/HR",
        "recommended_targets": ["attrition", "performance_rating", "promotion", "turnover"],
        "domain_tips": [
            "Attrition datasets are often imbalanced — most employees stay, few leave",
            "Watch for proxy variables like age or tenure that could introduce bias",
            "Retention impact matters more to HR stakeholders than raw accuracy",
        ],
    },
    "education": {
        "display_name": "Education",
        "icon": "🎓",
        "description": "This appears to be an academic or student performance dataset",
        "insight_language": "educational",
        "recommended_targets": ["grade", "score", "pass", "gpa", "result"],
        "domain_tips": [
            "Prior performance is often the strongest predictor of future outcomes",
            "Attendance and engagement features tend to correlate strongly with results",
            "Be cautious drawing causal conclusions from correlational student data",
        ],
    },
    "ecommerce": {
        "display_name": "E-commerce / Retail",
        "icon": "🛒",
        "description": "This appears to be an e-commerce or retail transactions dataset",
        "insight_language": "customer behavior",
        "recommended_targets": ["churn", "purchase", "return", "rating"],
        "domain_tips": [
            "Customer churn is usually imbalanced — most customers don't churn",
            "Recency, frequency, and monetary (RFM) features are strong predictors here",
            "Seasonality can heavily influence purchase and order patterns",
        ],
    },
    "marketing": {
        "display_name": "Marketing",
        "icon": "📣",
        "description": "This appears to be a marketing or campaign performance dataset",
        "insight_language": "conversion/marketing",
        "recommended_targets": ["conversion", "click", "lead", "churn"],
        "domain_tips": [
            "Conversion events are typically rare — expect class imbalance",
            "Check for leakage from post-conversion features when attributing results",
            "Segment-level analysis often reveals more than dataset-wide averages",
        ],
    },
    "general": {
        "display_name": "General Dataset",
        "icon": "📊",
        "description": "No strong domain signal was detected in the column names",
        "insight_language": "general data science",
        "recommended_targets": [],
        "domain_tips": [
            "Explore column names and values manually to understand the context",
            "Standard EDA practices apply — check missing values, outliers, and correlations",
        ],
    },
}


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
    @staticmethod
    def _keyword_matches_column(keyword: str, column_name: str) -> bool:
        # Whole-word match, but underscores/hyphens still count as word
        # separators (regex's own \b treats "_" as a word character, which
        # would stop "job" from matching "job_satisfaction" — snake_case
        # column names are the norm, so plain \b is too strict here).
        pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
        return re.search(pattern, column_name, re.IGNORECASE) is not None

    # ------------------------------------------------------------------
    def detect_domain(self, df: pd.DataFrame, profile: dict) -> dict:
        columns_lower = [c.lower() for c in df.columns]

        scores = {}
        matches = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matched = {
                kw for kw in keywords
                if any(self._keyword_matches_column(kw, col) for col in columns_lower)
            }
            scores[domain] = len(matched)
            matches[domain] = sorted(matched)

        best_count = max(scores.values()) if scores else 0
        top_domains = [d for d, s in scores.items() if s == best_count]

        # A single stray keyword match (e.g. a lone "age" column) isn't
        # strong enough evidence to call a domain — only 2+ matches (with
        # a single, unambiguous winner) count as a real signal.
        if best_count < 2 or len(top_domains) > 1:
            domain_key = "general"
            matched_keywords = []
            confidence = "none"
        else:
            domain_key = top_domains[0]
            matched_keywords = matches[domain_key]
            if best_count >= 4:
                confidence = "high"
            else:
                confidence = "medium"

        context = DOMAIN_CONTEXT[domain_key]

        return {
            "domain": domain_key,
            "display_name": context["display_name"],
            "icon": context["icon"],
            "confidence": confidence,
            "matched_keywords": matched_keywords,
            "description": context["description"],
            "domain_tips": context["domain_tips"],
            "recommended_targets": context["recommended_targets"],
            "insight_language": context["insight_language"],
        }

    # ------------------------------------------------------------------
    def generate_preprocessing_recommendations(
        self, df: pd.DataFrame, profile: dict, domain_info: dict = None,
        target_col: str = None, selected_models: list = None,
    ) -> dict:
        models_set = set(selected_models or [])
        id_like = set(profile.get("id_like_columns", []))
        constant_cols = set(profile.get("constant_columns", []))
        column_types = profile.get("column_types", {})
        missing = profile.get("missing", {})
        unique_counts = profile.get("unique_counts", {})
        rows = profile.get("rows", len(df))

        columns_out = {}
        columns_to_drop = 0
        columns_to_encode = 0
        columns_to_scale = 0
        warnings_count = 0

        for col in df.columns:
            dtype = column_types.get(col, "categorical")
            missing_pct = missing.get(col, 0.0)
            unique_count = unique_counts.get(col, int(df[col].nunique()))
            is_numeric = dtype == "numeric"

            entry = {
                "dtype": dtype,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "locked": False,
                "is_target": False,
                "is_id_like": False,
                "is_constant": False,
            }

            # ---- special / locked columns ----
            if target_col and col == target_col:
                entry["locked"] = True
                entry["is_target"] = True
                entry["recommendations"] = {
                    "target": {"action": "target", "reason": "This is your prediction target column."}
                }
                columns_out[col] = entry
                continue

            if col in id_like:
                entry["locked"] = True
                entry["is_id_like"] = True
                entry["recommendations"] = {
                    "missing": {
                        "action": "drop",
                        "reason": (
                            "This column has all unique values. It is likely an ID column "
                            "and will not help the model learn patterns."
                        ),
                        "options": [],
                        "warning": False,
                    }
                }
                columns_out[col] = entry
                columns_to_drop += 1
                continue

            if col in constant_cols:
                entry["locked"] = True
                entry["is_constant"] = True
                entry["recommendations"] = {
                    "missing": {
                        "action": "drop",
                        "reason": "This column has only one unique value and carries no information for the model.",
                        "options": [],
                        "warning": False,
                    }
                }
                columns_out[col] = entry
                columns_to_drop += 1
                continue

            # ---- normal columns ----
            recs = {}
            will_drop = False
            col_warning = False

            # missing values
            if missing_pct == 0:
                recs["missing"] = {
                    "action": "none",
                    "reason": "No missing values detected",
                    "options": NUMERIC_MISSING_OPTIONS if is_numeric else CATEGORICAL_MISSING_OPTIONS,
                    "warning": False,
                }
            elif missing_pct < 20:
                if is_numeric:
                    try:
                        skew = round(float(df[col].skew()), 2)
                    except Exception:
                        skew = 0.0
                    entry["skewness"] = skew
                    if abs(skew) > 1:
                        recs["missing"] = {
                            "action": "fill_median",
                            "reason": (
                                f"{missing_pct}% missing. Column is skewed (skew={skew}). "
                                "Median is more robust than mean for skewed distributions."
                            ),
                            "options": NUMERIC_MISSING_OPTIONS,
                            "warning": False,
                        }
                    else:
                        recs["missing"] = {
                            "action": "fill_mean",
                            "reason": (
                                f"{missing_pct}% missing. Column is approximately normal (skew={skew}). "
                                "Mean imputation is appropriate here."
                            ),
                            "options": NUMERIC_MISSING_OPTIONS,
                            "warning": False,
                        }
                else:
                    recs["missing"] = {
                        "action": "fill_mode",
                        "reason": (
                            f"Categorical column with {missing_pct}% missing. Mode (most frequent value) "
                            "is the standard imputation strategy."
                        ),
                        "options": CATEGORICAL_MISSING_OPTIONS,
                        "warning": False,
                    }
            elif missing_pct < 50:
                col_warning = True
                fill_kind = "median" if is_numeric else "mode"
                recs["missing"] = {
                    "action": "fill_median" if is_numeric else "fill_mode",
                    "reason": (
                        f"High missing rate ({missing_pct}%). Consider whether this column is reliable. "
                        f"Imputing with {fill_kind} for now."
                    ),
                    "options": NUMERIC_MISSING_OPTIONS if is_numeric else CATEGORICAL_MISSING_OPTIONS,
                    "warning": True,
                }
            else:
                will_drop = True
                col_warning = True
                recs["missing"] = {
                    "action": "drop",
                    "reason": (
                        f"{missing_pct}% of values are missing. Dropping this column as imputation "
                        "would introduce more noise than signal."
                    ),
                    "options": NUMERIC_MISSING_OPTIONS if is_numeric else CATEGORICAL_MISSING_OPTIONS,
                    "warning": True,
                }

            if will_drop:
                columns_to_drop += 1

            # encoding (categorical only)
            if not is_numeric:
                if will_drop:
                    recs["encoding"] = {"action": "none", "reason": "Column will be dropped."}
                else:
                    if unique_count == 2:
                        recs["encoding"] = {
                            "action": "binary_encode",
                            "reason": "Only 2 unique values. Binary encoding (0/1) is the most efficient choice.",
                            "options": ["binary_encode", "label_encode"],
                            "warning": False,
                        }
                        columns_to_encode += 1
                    elif unique_count <= 10:
                        recs["encoding"] = {
                            "action": "onehot_encode",
                            "reason": (
                                f"{unique_count} unique categories. One-hot encoding avoids imposing "
                                "ordinal relationships between categories."
                            ),
                            "options": ["onehot_encode", "label_encode", "binary_encode"],
                            "warning": False,
                        }
                        columns_to_encode += 1
                    elif unique_count <= 50:
                        recs["encoding"] = {
                            "action": "label_encode",
                            "reason": (
                                f"{unique_count} unique categories is too many for one-hot encoding "
                                f"(would create {unique_count} new columns). Label encoding is more practical here."
                            ),
                            "options": ["label_encode", "onehot_encode"],
                            "warning": False,
                        }
                        columns_to_encode += 1
                    else:
                        col_warning = True
                        recs["encoding"] = {
                            "action": "label_encode",
                            "reason": (
                                f"High cardinality ({unique_count} unique values). Label encoding recommended. "
                                "Consider dropping this column if it is an ID."
                            ),
                            "options": ["label_encode", "drop"],
                            "warning": True,
                        }
                        columns_to_encode += 1

            # outliers (numeric only)
            if is_numeric:
                try:
                    series = df[col].dropna()
                    if len(series) > 0:
                        q1, q3 = series.quantile(0.25), series.quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        outlier_pct = round(float(((series < lower) | (series > upper)).mean() * 100), 2)
                    else:
                        outlier_pct = 0.0
                except Exception:
                    outlier_pct = 0.0

                entry["outlier_pct"] = outlier_pct

                if will_drop:
                    recs["outliers"] = {"action": "none", "reason": "Column will be dropped."}
                elif outlier_pct == 0:
                    recs["outliers"] = {
                        "action": "keep", "reason": "No outliers detected.",
                        "options": OUTLIER_OPTIONS_DEFAULT, "warning": False,
                    }
                elif outlier_pct <= 5:
                    recs["outliers"] = {
                        "action": "keep",
                        "reason": f"Few outliers ({outlier_pct}%). Not significant enough to require treatment.",
                        "options": OUTLIER_OPTIONS_DEFAULT, "warning": False,
                    }
                elif outlier_pct <= 20:
                    if models_set & OUTLIER_SENSITIVE_MODELS:
                        recs["outliers"] = {
                            "action": "cap_iqr",
                            "reason": (
                                f"{outlier_pct}% outliers detected. Capping recommended because your "
                                "selected models (SVM/KNN) are sensitive to outliers."
                            ),
                            "options": OUTLIER_OPTIONS_DEFAULT, "warning": False,
                        }
                    else:
                        recs["outliers"] = {
                            "action": "keep",
                            "reason": (
                                f"{outlier_pct}% outliers detected. Tree-based models like Random Forest "
                                "handle outliers naturally."
                            ),
                            "options": OUTLIER_OPTIONS_DEFAULT, "warning": False,
                        }
                else:
                    col_warning = True
                    recs["outliers"] = {
                        "action": "cap_iqr",
                        "reason": (
                            f"High outlier rate ({outlier_pct}%). Capping at IQR boundaries recommended "
                            "to prevent model distortion."
                        ),
                        "options": OUTLIER_OPTIONS_HIGH, "warning": True,
                    }

            # scaling (numeric only)
            if is_numeric:
                if will_drop:
                    recs["scaling"] = {"action": "none", "reason": "Column will be dropped."}
                else:
                    if models_set & SCALE_SENSITIVE_MODELS:
                        action = "standard_scale"
                        reason = (
                            "Your selected models are sensitive to feature scale. "
                            "StandardScaler will normalize this column."
                        )
                    elif models_set and models_set.issubset(SCALE_INVARIANT_MODELS):
                        action = "none"
                        reason = "Tree-based models are scale-invariant. Scaling is not required."
                    else:
                        action = "standard_scale"
                        reason = "Scaling recommended as a safe default."
                    recs["scaling"] = {"action": action, "reason": reason, "options": SCALING_OPTIONS}
                    if action != "none":
                        columns_to_scale += 1

            entry["recommendations"] = recs
            if col_warning:
                warnings_count += 1
            columns_out[col] = entry

        summary = {
            "total_columns": len(df.columns),
            "columns_to_drop": columns_to_drop,
            "columns_to_encode": columns_to_encode,
            "columns_to_scale": columns_to_scale,
            "warnings": warnings_count,
        }

        return {"columns": columns_out, "summary": summary}

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
    def generate_insights(self, df, profile, outliers, target_info, problem_type, domain_info: dict = None) -> list:
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

        domain_key = (domain_info or {}).get("domain", "general")
        domain_display = (domain_info or {}).get("display_name", "General Dataset")
        domain_instruction = ""
        if domain_key != "general":
            domain_instruction = (
                f"\nThis is a {domain_display} dataset.\n"
                f"Frame your insights using {domain_key} terminology.\n"
                "For example:\n"
                "- Healthcare: mention clinical significance\n"
                "- Finance: mention risk and regulatory aspects\n"
                "- HR: mention business impact on retention\n"
                "- Education: mention learning outcomes\n"
                "- E-commerce: mention customer behavior\n"
                "- Marketing: mention conversion impact\n"
            )

        prompt = (
            f"Dataset shape: {shape_str}\n"
            f"Problem type: {problem_type.get('problem_type')}\n"
            f"Missing values: {missing_str}\n"
            f"Outliers: {outlier_str}\n"
            f"Target balance: {balance_str}\n"
            f"Strong correlations: {corr_str}\n"
            f"Data quality score: {profile['quality_score']}/100\n"
            f"{domain_instruction}\n"
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
        domain_info = self.detect_domain(df, profile)

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

        insights = self.generate_insights(df, profile, outliers, target_balance, problem_type, domain_info)

        return {
            "profile": profile,
            "domain_info": domain_info,
            "target_suggestion": target_suggestion,
            "problem_type": problem_type,
            "charts": charts,
            "eda_code_path": eda_code_path,
            "outliers": outliers,
            "target_balance": target_balance,
            "insights": insights,
            "status": "completed",
        }
