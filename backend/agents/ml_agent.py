# ============================================
# ZeroCode — ml_agent.py
# ML Agent — handles data preprocessing,
# automatic model selection, training,
# evaluation, and saving the best model.
# ============================================

import json
import os
import pickle
import time

import kaleido
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix as sk_confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error as sk_max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

from config import UPLOAD_FOLDER
from llm_provider import LLMProvider

CLASSIFICATION_TYPES = ("binary_classification", "multiclass_classification")

CHARTS_FOLDER = os.path.join(UPLOAD_FOLDER, "charts")

SHAP_TREE_MODELS = {
    "Random Forest", "Random Forest Regressor",
    "Extra Trees", "Decision Tree", "Decision Tree Regressor",
    "AdaBoost", "XGBoost", "XGBoost Regressor",
    "LightGBM", "LightGBM Regressor", "CatBoost", "CatBoost Regressor",
}
SHAP_LINEAR_MODELS = {
    "Logistic Regression", "Linear Regression", "Ridge", "Lasso", "ElasticNet",
}

CORE_CLASSIFICATION_METRICS = ["accuracy", "f1", "precision", "recall"]
CORE_REGRESSION_METRICS = ["mae", "rmse", "r2"]
EXTENDED_CLASSIFICATION_METRICS = [
    "roc_auc", "log_loss", "mcc", "cohen_kappa", "balanced_accuracy", "confusion_matrix",
]
EXTENDED_REGRESSION_METRICS = [
    "mape", "explained_variance", "max_error", "median_ae", "adjusted_r2",
]

MODEL_REASON_FALLBACKS = {
    "Logistic Regression": "Simple, fast linear baseline that's easy to interpret.",
    "Linear Regression": "Simple, fast linear baseline that's easy to interpret.",
    "Random Forest": "Robust general-purpose model, handles mixed features and outliers well.",
    "Random Forest Regressor": "Robust general-purpose model, handles mixed features and outliers well.",
    "XGBoost": "Strong gradient boosting performance, especially on larger datasets.",
    "XGBoost Regressor": "Strong gradient boosting performance, especially on larger datasets.",
    "SVM": "Effective on smaller datasets with clear separation between classes.",
    "SVR": "Effective on smaller datasets with clear non-linear patterns.",
    "Decision Tree": "Simple, interpretable baseline to compare other models against.",
    "Decision Tree Regressor": "Simple, interpretable baseline to compare other models against.",
    "KNN": "Distance-based model that works best on smaller, low-dimensional data.",
    "KNN Regressor": "Distance-based model that works best on smaller, low-dimensional data.",
    "AdaBoost": "Boosting approach that can help correct for imbalanced classes.",
    "AdaBoost Regressor": "Boosting approach that focuses on hard-to-predict samples.",
}

SCALE_SENSITIVE_MODELS = {"SVM", "KNN", "Logistic Regression"}
SCALE_INVARIANT_MODELS = {"Random Forest", "XGBoost"}
OUTLIER_SENSITIVE_MODELS = {"SVM", "KNN"}

PARAM_GRIDS = {
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "Random Forest Regressor": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "XGBoost": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]},
    "XGBoost Regressor": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "Decision Tree": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "Decision Tree Regressor": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
}

MODEL_IMPORTS = {
    "Random Forest": ("sklearn.ensemble", "RandomForestClassifier"),
    "Random Forest Regressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "Logistic Regression": ("sklearn.linear_model", "LogisticRegression"),
    "Linear Regression": ("sklearn.linear_model", "LinearRegression"),
    "Decision Tree": ("sklearn.tree", "DecisionTreeClassifier"),
    "Decision Tree Regressor": ("sklearn.tree", "DecisionTreeRegressor"),
    "SVM": ("sklearn.svm", "SVC"),
    "SVR": ("sklearn.svm", "SVR"),
    "XGBoost": ("xgboost", "XGBClassifier"),
    "XGBoost Regressor": ("xgboost", "XGBRegressor"),
    "KNN": ("sklearn.neighbors", "KNeighborsClassifier"),
    "AdaBoost": ("sklearn.ensemble", "AdaBoostClassifier"),
    "Extra Trees": ("sklearn.ensemble", "ExtraTreesClassifier"),
    "Naive Bayes": ("sklearn.naive_bayes", "GaussianNB"),
    "LightGBM": ("lightgbm", "LGBMClassifier"),
    "CatBoost": ("catboost", "CatBoostClassifier"),
    "KNN Regressor": ("sklearn.neighbors", "KNeighborsRegressor"),
    "Lasso": ("sklearn.linear_model", "Lasso"),
    "Ridge": ("sklearn.linear_model", "Ridge"),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet"),
    "LightGBM Regressor": ("lightgbm", "LGBMRegressor"),
    "CatBoost Regressor": ("catboost", "CatBoostRegressor"),
}


class MLAgent:
    def __init__(self):
        self.llm = LLMProvider()

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_missing_action(series: pd.Series, is_numeric: bool) -> str:
        if is_numeric:
            try:
                skew = float(series.skew())
            except Exception:
                skew = 0.0
            return "fill_median" if abs(skew) > 1 else "fill_mean"
        return "fill_mode"

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_encoding_action(nunique: int) -> str:
        if nunique == 2:
            return "binary_encode"
        if nunique <= 10:
            return "onehot_encode"
        return "label_encode"

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_outlier_action(outlier_pct: float, selected_models: set) -> str:
        if outlier_pct <= 5:
            return "keep"
        if outlier_pct <= 20:
            return "cap_iqr" if selected_models & OUTLIER_SENSITIVE_MODELS else "keep"
        return "cap_iqr"

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_scaling_action(selected_models: set) -> str:
        if selected_models & SCALE_SENSITIVE_MODELS:
            return "standard_scale"
        if selected_models and selected_models.issubset(SCALE_INVARIANT_MODELS):
            return "none"
        return "standard_scale"

    # ------------------------------------------------------------------
    def preprocess(
        self, df: pd.DataFrame, target_col: str, problem_type: str,
        user_choices: dict = None, selected_models: list = None,
    ) -> dict:
        log = []
        df = df.copy()
        rows = len(df)
        user_choices = user_choices or {}
        models_set = set(selected_models or [])

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # b) drop useless columns (locked — never user-overridable)
        for col in list(X.columns):
            nunique = X[col].nunique()
            if rows > 0 and nunique == rows:
                X = X.drop(columns=[col])
                log.append(f"Dropped '{col}' column — all unique values (ID-like)")
            elif nunique == 1:
                X = X.drop(columns=[col])
                log.append(f"Dropped '{col}' column — constant value")

        # c) missing values
        for col in list(X.columns):
            missing_pct = round(float(X[col].isna().mean() * 100), 2)
            is_numeric = pd.api.types.is_numeric_dtype(X[col])
            auto_action = "none" if missing_pct == 0 else (
                "drop" if missing_pct >= 50 else self._auto_missing_action(X[col], is_numeric)
            )
            user_action = (user_choices.get(col) or {}).get("missing")
            action = user_action or auto_action

            if user_action and user_action != auto_action:
                log.append(f"User override: {col} → {action} (AI suggested: {auto_action})")

            if action == "none":
                continue
            if action == "drop":
                X = X.drop(columns=[col])
                log.append(f"Dropped '{col}' — {missing_pct}% missing")
                continue
            if action == "fill_median":
                fill_val = X[col].median()
                X[col] = X[col].fillna(fill_val)
                log.append(f"Filled '{col}' with median ({fill_val})")
            elif action == "fill_mean":
                fill_val = X[col].mean()
                X[col] = X[col].fillna(fill_val)
                log.append(f"Filled '{col}' with mean ({fill_val})")
            elif action == "fill_zero":
                X[col] = X[col].fillna(0)
                log.append(f"Filled '{col}' with 0")
            elif action == "fill_mode":
                mode_series = X[col].mode()
                mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                X[col] = X[col].fillna(mode_val)
                log.append(f"Filled '{col}' with mode ('{mode_val}')")
            elif action == "fill_unknown":
                X[col] = X[col].fillna("Unknown")
                log.append(f"Filled '{col}' with 'Unknown'")

        # Capture genuinely-numeric columns BEFORE encoding — label/one-hot/binary
        # encoding produces new numeric-dtype columns that are nominal codes, not
        # continuous values, so outlier detection/capping and scaling must only
        # ever consider the original numeric columns, not post-encoding artifacts.
        original_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # d) encode categorical columns
        label_encoders = {}
        for col in list(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
                continue

            nunique = X[col].nunique()
            auto_action = self._auto_encoding_action(nunique)
            user_action = (user_choices.get(col) or {}).get("encoding")
            action = user_action or auto_action

            if user_action and user_action != auto_action:
                log.append(f"User override: {col} → {action} (AI suggested: {auto_action})")

            if action == "drop":
                X = X.drop(columns=[col])
                log.append(f"Dropped '{col}' column per encoding choice")
            elif action == "binary_encode":
                uniques = sorted(X[col].dropna().unique().tolist(), key=str)
                if len(uniques) < 2:
                    uniques = uniques + [uniques[0] if uniques else "Unknown"]
                mapping = {uniques[0]: 0, uniques[1]: 1}
                X[col] = X[col].map(mapping).fillna(0)
                log.append(f"Binary encoded '{col}' ({mapping})")
            elif action == "onehot_encode":
                dummies = pd.get_dummies(X[col], prefix=col).astype(int)
                X = X.drop(columns=[col])
                X = pd.concat([X, dummies], axis=1)
                log.append(f"One-Hot encoded '{col}' ({nunique} categories)")
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                log.append(f"Label encoded '{col}' ({nunique} categories)")

        # e) outlier handling (original numeric columns only — see note above)
        for col in [c for c in original_numeric_cols if c in X.columns]:
            series = X[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_pct = round(float(((series < lower) | (series > upper)).mean() * 100), 2)
            if outlier_pct == 0:
                continue

            auto_action = self._auto_outlier_action(outlier_pct, models_set)
            user_action = (user_choices.get(col) or {}).get("outliers")
            action = user_action or auto_action

            if user_action and user_action != auto_action:
                log.append(f"User override: {col} → {action} (AI suggested: {auto_action})")

            if action == "cap_iqr":
                X[col] = X[col].clip(lower=lower, upper=upper)
                log.append(f"Capped '{col}' outliers to IQR bounds ({outlier_pct}% affected)")
            elif action == "log_transform":
                shift = abs(min(X[col].min(), 0)) + 1
                X[col] = np.log1p(X[col] + shift)
                log.append(f"Applied log transform to '{col}' ({outlier_pct}% outliers)")

        # f) scale numeric features — grouped per-column choice (standard/minmax/none)
        numeric_cols = [c for c in original_numeric_cols if c in X.columns]
        standard_cols, minmax_cols = [], []
        for col in numeric_cols:
            auto_action = self._auto_scaling_action(models_set)
            user_action = (user_choices.get(col) or {}).get("scaling")
            action = user_action or auto_action

            if user_action and user_action != auto_action:
                log.append(f"User override: {col} → {action} (AI suggested: {auto_action})")

            if action == "standard_scale":
                standard_cols.append(col)
            elif action == "minmax_scale":
                minmax_cols.append(col)

        standard_scaler = None
        if standard_cols:
            standard_scaler = StandardScaler()
            X[standard_cols] = standard_scaler.fit_transform(X[standard_cols])
            log.append(f"Applied StandardScaler to {len(standard_cols)} column(s): {standard_cols}")

        minmax_scaler = None
        if minmax_cols:
            minmax_scaler = MinMaxScaler()
            X[minmax_cols] = minmax_scaler.fit_transform(X[minmax_cols])
            log.append(f"Applied MinMaxScaler to {len(minmax_cols)} column(s): {minmax_cols}")

        scaler = {
            "standard": standard_scaler,
            "minmax": minmax_scaler,
            "standard_cols": standard_cols,
            "minmax_cols": minmax_cols,
        }

        # encode target if classification and non-numeric
        target_encoder = None
        if problem_type in CLASSIFICATION_TYPES and not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index, name=target_col)
            log.append(f"Label encoded target column '{target_col}'")

        # f) handle imbalanced target (classification only)
        use_class_weight = False
        if problem_type in CLASSIFICATION_TYPES:
            value_counts = y.value_counts(normalize=True)
            if (value_counts < 0.30).any():
                use_class_weight = True
                log.append("Detected class imbalance (<30% representation) — using class_weight='balanced'")

        # save the fully cleaned dataset and a standalone reproduction script
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        cleaned_df = X.copy()
        cleaned_df[target_col] = y
        cleaned_data_path = os.path.join(UPLOAD_FOLDER, "preprocessed_data.csv")
        cleaned_df.to_csv(cleaned_data_path, index=False, encoding="utf-8")
        log.append("Saved preprocessed_data.csv to outputs/")

        preprocessing_code_path = os.path.join(UPLOAD_FOLDER, "preprocessing.py")
        with open(preprocessing_code_path, "w", encoding="utf-8") as f:
            f.write(self._generate_preprocessing_script(log, target_col))
        log.append("Saved preprocessing.py to outputs/")

        # g) split data
        stratify = y if problem_type in CLASSIFICATION_TYPES else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            log.append("Stratified split failed (a class has fewer than 2 members) — used random split instead")

        log.append(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": X.columns.tolist(),
            "preprocessing_log": log,
            "use_class_weight": use_class_weight,
            "label_encoders": label_encoders,
            "target_encoder": target_encoder,
            "scaler": scaler,
            "preprocessed_data_path": cleaned_data_path.replace("\\", "/"),
            "preprocessing_code_path": preprocessing_code_path.replace("\\", "/"),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_preprocessing_script(preprocessing_log: list, target_col: str) -> str:
        log_comments = "\n".join(f"# - {entry}" for entry in preprocessing_log) or "# (no preprocessing steps recorded)"

        template = '''# Auto-generated ZeroCode preprocessing script
# Reproduces the cleaning, encoding, and scaling steps applied to this dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------------------------
# Load your dataset here
DATA_PATH = "your_dataset.csv"
df = pd.read_csv(DATA_PATH)

TARGET_COLUMN = "__TARGET_COL__"

# ------------------------------------------------------------------
# Steps applied by ZeroCode on this dataset:
__LOG_COMMENTS__

y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN])
rows = len(df)

# ------------------------------------------------------------------
# Drop ID-like and constant columns
for col in list(X.columns):
    nunique = X[col].nunique()
    if rows > 0 and nunique == rows:
        X = X.drop(columns=[col])
    elif nunique == 1:
        X = X.drop(columns=[col])

# ------------------------------------------------------------------
# Handle missing values
for col in list(X.columns):
    missing_pct = X[col].isna().mean() * 100
    if missing_pct > 50:
        X = X.drop(columns=[col])
        continue
    if missing_pct > 0:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            mode_series = X[col].mode()
            X[col] = X[col].fillna(mode_series.iloc[0] if not mode_series.empty else "Unknown")

# ------------------------------------------------------------------
# Encode categorical columns
for col in list(X.columns):
    if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
        continue
    nunique = X[col].nunique()
    if nunique == 2:
        uniques = sorted(X[col].dropna().unique().tolist(), key=str)
        X[col] = X[col].map({uniques[0]: 0, uniques[1]: 1})
    elif nunique < 10:
        dummies = pd.get_dummies(X[col], prefix=col).astype(int)
        X = X.drop(columns=[col])
        X = pd.concat([X, dummies], axis=1)
    else:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# ------------------------------------------------------------------
# Scale numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
if numeric_cols:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ------------------------------------------------------------------
# Encode target if needed
if not pd.api.types.is_numeric_dtype(y):
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index, name=TARGET_COLUMN)

# ------------------------------------------------------------------
# Save cleaned dataset
cleaned_df = X.copy()
cleaned_df[TARGET_COLUMN] = y
cleaned_df.to_csv("preprocessed_data.csv", index=False)
print("Saved preprocessed_data.csv with shape:", cleaned_df.shape)
'''

        return template.replace("__TARGET_COL__", target_col).replace("__LOG_COMMENTS__", log_comments)

    # ------------------------------------------------------------------
    def select_models(self, problem_type: str, df: pd.DataFrame, use_class_weight: bool) -> dict:
        rows = len(df)
        class_weight = "balanced" if use_class_weight else None
        selected_models = {}
        reasoning = {}

        if problem_type in CLASSIFICATION_TYPES:
            selected_models["Random Forest"] = RandomForestClassifier(
                n_estimators=100, class_weight=class_weight, random_state=42
            )
            reasoning["Random Forest"] = "Handles mixed features well, robust to outliers"

            selected_models["Logistic Regression"] = LogisticRegression(
                class_weight=class_weight, max_iter=1000
            )
            reasoning["Logistic Regression"] = "Strong baseline for classification problems"

            selected_models["Decision Tree"] = DecisionTreeClassifier(
                class_weight=class_weight, random_state=42
            )
            reasoning["Decision Tree"] = "Simple interpretable baseline model"

            if rows > 1000 and XGBClassifier is not None:
                selected_models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)
                reasoning["XGBoost"] = "High performance on medium-large datasets"
            else:
                selected_models["SVM"] = SVC(class_weight=class_weight)
                reasoning["SVM"] = "Effective for small datasets with clear margins"

        else:
            selected_models["Random Forest Regressor"] = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
            reasoning["Random Forest Regressor"] = "Handles non-linear relationships well, robust to outliers"

            selected_models["Linear Regression"] = LinearRegression()
            reasoning["Linear Regression"] = "Strong interpretable baseline for regression problems"

            selected_models["Decision Tree Regressor"] = DecisionTreeRegressor(random_state=42)
            reasoning["Decision Tree Regressor"] = "Simple interpretable baseline model"

            if rows > 1000 and XGBRegressor is not None:
                selected_models["XGBoost Regressor"] = XGBRegressor(random_state=42)
                reasoning["XGBoost Regressor"] = "High performance on medium-large datasets"
            else:
                selected_models["SVR"] = SVR()
                reasoning["SVR"] = "Effective for small datasets with non-linear patterns"

        return {"selected_models": selected_models, "reasoning": reasoning}

    # ------------------------------------------------------------------
    @staticmethod
    def get_model_from_name(name: str, problem_type: str, use_class_weight: bool):
        class_weight = "balanced" if use_class_weight else None

        if problem_type in CLASSIFICATION_TYPES:
            factories = {
                "Logistic Regression": lambda: LogisticRegression(class_weight=class_weight, max_iter=1000),
                "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42),
                "XGBoost": lambda: XGBClassifier(eval_metric="mlogloss", random_state=42) if XGBClassifier is not None else None,
                "SVM": lambda: SVC(class_weight=class_weight),
                "Decision Tree": lambda: DecisionTreeClassifier(class_weight=class_weight, random_state=42),
                "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
                "AdaBoost": lambda: AdaBoostClassifier(),
                "Extra Trees": lambda: ExtraTreesClassifier(class_weight=class_weight, random_state=42),
                "Naive Bayes": lambda: GaussianNB(),
                "LightGBM": lambda: LGBMClassifier() if LGBMClassifier is not None else None,
                "CatBoost": lambda: CatBoostClassifier(verbose=0) if CatBoostClassifier is not None else None,
            }
        else:
            factories = {
                "Linear Regression": lambda: LinearRegression(),
                "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
                "XGBoost Regressor": lambda: XGBRegressor(random_state=42) if XGBRegressor is not None else None,
                "SVR": lambda: SVR(),
                "Decision Tree Regressor": lambda: DecisionTreeRegressor(random_state=42),
                "KNN Regressor": lambda: KNeighborsRegressor(),
                "AdaBoost Regressor": lambda: AdaBoostRegressor(),
                "Lasso": lambda: Lasso(),
                "Ridge": lambda: Ridge(),
                "ElasticNet": lambda: ElasticNet(),
                "LightGBM Regressor": lambda: LGBMRegressor() if LGBMRegressor is not None else None,
                "CatBoost Regressor": lambda: CatBoostRegressor(verbose=0) if CatBoostRegressor is not None else None,
            }

        factory = factories.get(name)
        if factory is None:
            return None

        return factory()

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_metric(fn, round_digits=4):
        try:
            value = fn()
            if value is None:
                return None
            if round_digits is None:
                return value
            return round(float(value), round_digits)
        except Exception:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mape(y_true, y_pred):
        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            mask = y_true_arr != 0
            if mask.sum() == 0:
                return None
            pct_errors = np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])
            return round(float(np.mean(pct_errors) * 100), 4)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def train_models(self, selected_models, X_train, X_test, y_train, y_test, problem_type: str, selected_metrics: list = None) -> dict:
        is_classification = problem_type in CLASSIFICATION_TYPES
        extended = set(selected_metrics or [])
        results = {}

        for name, model in selected_models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                elapsed = time.time() - start
                preds = model.predict(X_test)

                metrics = {}

                if is_classification:
                    metrics["accuracy"] = self._safe_metric(lambda: accuracy_score(y_test, preds))
                    metrics["f1"] = self._safe_metric(lambda: f1_score(y_test, preds, average="weighted", zero_division=0))
                    metrics["precision"] = self._safe_metric(lambda: precision_score(y_test, preds, average="weighted", zero_division=0))
                    metrics["recall"] = self._safe_metric(lambda: recall_score(y_test, preds, average="weighted", zero_division=0))

                    proba = None
                    if "roc_auc" in extended or "log_loss" in extended:
                        if hasattr(model, "predict_proba"):
                            try:
                                proba = model.predict_proba(X_test)
                            except Exception:
                                proba = None

                    if "roc_auc" in extended:
                        def _roc_auc():
                            if proba is None:
                                raise ValueError("predict_proba not available")
                            if proba.shape[1] == 2:
                                return roc_auc_score(y_test, proba[:, 1])
                            return roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
                        metrics["roc_auc"] = self._safe_metric(_roc_auc)

                    if "log_loss" in extended:
                        def _log_loss():
                            if proba is None:
                                raise ValueError("predict_proba not available")
                            return log_loss(y_test, proba)
                        metrics["log_loss"] = self._safe_metric(_log_loss)

                    if "mcc" in extended:
                        metrics["mcc"] = self._safe_metric(lambda: matthews_corrcoef(y_test, preds))

                    if "cohen_kappa" in extended:
                        metrics["cohen_kappa"] = self._safe_metric(lambda: cohen_kappa_score(y_test, preds))

                    if "balanced_accuracy" in extended:
                        metrics["balanced_accuracy"] = self._safe_metric(lambda: balanced_accuracy_score(y_test, preds))

                    if "confusion_matrix" in extended:
                        metrics["confusion_matrix"] = self._safe_metric(
                            lambda: sk_confusion_matrix(y_test, preds).tolist(), round_digits=None
                        )

                else:
                    mae = mean_absolute_error(y_test, preds)
                    rmse = mean_squared_error(y_test, preds) ** 0.5
                    r2 = r2_score(y_test, preds)
                    metrics["mae"] = round(float(mae), 4)
                    metrics["rmse"] = round(float(rmse), 4)
                    metrics["r2"] = round(float(r2), 4)

                    if "mape" in extended:
                        metrics["mape"] = self._safe_mape(y_test, preds)

                    if "explained_variance" in extended:
                        metrics["explained_variance"] = self._safe_metric(lambda: explained_variance_score(y_test, preds))

                    if "max_error" in extended:
                        metrics["max_error"] = self._safe_metric(lambda: sk_max_error(y_test, preds))

                    if "median_ae" in extended:
                        metrics["median_ae"] = self._safe_metric(lambda: median_absolute_error(y_test, preds))

                    if "adjusted_r2" in extended:
                        def _adjusted_r2():
                            n = len(y_test)
                            k = X_test.shape[1]
                            if n - k - 1 <= 0:
                                return None
                            return 1 - (1 - r2) * (n - 1) / (n - k - 1)
                        metrics["adjusted_r2"] = self._safe_metric(_adjusted_r2)

                results[name] = {
                    "model": model,
                    "metrics": metrics,
                    "training_time": f"{elapsed:.2f}s",
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        return {"results": results}

    # ------------------------------------------------------------------
    def tune_best_model(self, best_model_name: str, best_model, X_train, y_train, problem_type: str) -> dict:
        param_grid = PARAM_GRIDS.get(best_model_name)
        if not param_grid:
            return {"tuned_model": best_model, "best_params": {}, "tuning_score": None}

        scoring = "r2" if problem_type == "regression" else "accuracy"

        try:
            search = RandomizedSearchCV(
                best_model,
                param_distributions=param_grid,
                n_iter=10,
                cv=3,
                n_jobs=-1,
                scoring=scoring,
                random_state=42,
            )
            search.fit(X_train, y_train)
            return {
                "tuned_model": search.best_estimator_,
                "best_params": search.best_params_,
                "tuning_score": round(float(search.best_score_), 4),
            }
        except Exception:
            return {"tuned_model": best_model, "best_params": {}, "tuning_score": None}

    # ------------------------------------------------------------------
    def generate_leaderboard(self, results: dict, problem_type: str) -> dict:
        metric_key = "r2" if problem_type == "regression" else "f1"
        valid_results = {name: r for name, r in results.items() if "error" not in r}

        def sort_key(name):
            value = valid_results[name].get("metrics", {}).get(metric_key)
            return value if value is not None else float("-inf")

        sorted_names = sorted(valid_results.keys(), key=sort_key, reverse=True)

        leaderboard = []
        for i, name in enumerate(sorted_names):
            r = valid_results[name]
            entry = {
                "rank": i + 1,
                "model": name,
                "training_time": r["training_time"],
                "is_best": i == 0,
            }
            entry.update(r.get("metrics", {}))
            leaderboard.append(entry)

        best_model_name = sorted_names[0] if sorted_names else None
        return {"leaderboard": leaderboard, "best_model_name": best_model_name}

    # ------------------------------------------------------------------
    def save_model(self, tuned_model, best_model_name: str, feature_names: list, scaler) -> dict:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        model_path = os.path.join(UPLOAD_FOLDER, "best_model.pkl")
        feature_names_path = os.path.join(UPLOAD_FOLDER, "feature_names.json")
        scaler_path = os.path.join(UPLOAD_FOLDER, "scaler.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(tuned_model, f)

        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        return {
            "model_path": model_path.replace("\\", "/"),
            "feature_names_path": feature_names_path.replace("\\", "/"),
            "scaler_path": scaler_path.replace("\\", "/"),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _score_candidate_models(
        rows, numeric_features, categorical_features, has_outliers, is_imbalanced, size_category, is_regression
    ) -> dict:
        total_features = numeric_features + categorical_features
        few_categorical = categorical_features <= 3
        few_features = total_features <= 6
        many_features = total_features > 10

        if is_regression:
            names = {
                "linear": "Linear Regression",
                "forest": "Random Forest Regressor",
                "xgb": "XGBoost Regressor",
                "svm": "SVR",
                "tree": "Decision Tree Regressor",
                "knn": "KNN Regressor",
                "ada": "AdaBoost Regressor",
            }
            xgb_available = XGBRegressor is not None
        else:
            names = {
                "linear": "Logistic Regression",
                "forest": "Random Forest",
                "xgb": "XGBoost",
                "svm": "SVM",
                "tree": "Decision Tree",
                "knn": "KNN",
                "ada": "AdaBoost",
            }
            xgb_available = XGBClassifier is not None

        scores = {}

        s = 0
        if rows < 5000:
            s += 2
        if few_categorical:
            s += 2
        if has_outliers:
            s -= 1
        scores[names["linear"]] = {"score": s, "tag": "Baseline Model"}

        s = 3
        if numeric_features > 0 and categorical_features > 0:
            s += 2
        if has_outliers:
            s += 1
        scores[names["forest"]] = {"score": s, "tag": "Recommended"}

        if xgb_available:
            s = 0
            if rows > 1000:
                s += 3
            if is_imbalanced:
                s += 2
            if many_features:
                s += 2
            scores[names["xgb"]] = {"score": s, "tag": "Recommended" if s > 4 else None}

        s = 0
        if rows < 2000:
            s += 3
        if rows > 5000:
            s -= 2
        if few_features:
            s += 1
        scores[names["svm"]] = {"score": s, "tag": None}

        scores[names["tree"]] = {"score": 1, "tag": "Baseline Model"}

        s = 0
        if rows < 3000:
            s += 2
        if rows > 5000:
            s -= 2
        if many_features:
            s -= 1
        scores[names["knn"]] = {"score": s, "tag": None}

        s = 0
        if is_imbalanced:
            s += 2
        if size_category == "medium":
            s += 1
        scores[names["ada"]] = {"score": s, "tag": None}

        return scores

    # ------------------------------------------------------------------
    def _get_model_reasons(
        self, model_names, rows, columns, problem_type, size_category,
        has_outliers, is_imbalanced, numeric_features, categorical_features
    ) -> dict:
        model_list_str = ", ".join(model_names)
        prompt = (
            "You are a data science expert.\n"
            "Given this dataset:\n"
            f"- Rows: {rows}\n"
            f"- Columns: {columns}\n"
            f"- Problem: {problem_type}\n"
            f"- Data size: {size_category}\n"
            f"- Has outliers: {'yes' if has_outliers else 'no'}\n"
            f"- Target imbalanced: {'yes' if is_imbalanced else 'no'}\n"
            f"- Feature types: {numeric_features} numeric, {categorical_features} categorical\n\n"
            "For each of these models explain in ONE short sentence (max 15 words) why it "
            "is or isn't suitable for this specific dataset:\n"
            f"{model_list_str}\n\n"
            "Return JSON only, an object mapping each model name exactly as given to its one-sentence reason."
        )
        system_prompt = "Respond with valid JSON only: an object whose keys are exactly the given model names."

        try:
            result = self.llm.ask_json(prompt, system_prompt=system_prompt)
        except Exception:
            result = {}

        if isinstance(result, dict) and result:
            return {name: reason for name, reason in result.items() if isinstance(reason, str)}
        return {}

    # ------------------------------------------------------------------
    def generate_model_recommendations(self, df: pd.DataFrame, problem_type: str, profile: dict, analysis_result: dict) -> dict:
        is_regression = problem_type == "regression"
        rows = df.shape[0]

        target_col = (analysis_result or {}).get("target_suggestion", {}).get("suggested_target")
        id_like_columns = set(profile.get("id_like_columns", []))
        column_types = profile.get("column_types", {})
        feature_types = {
            col: t for col, t in column_types.items()
            if col != target_col and col not in id_like_columns
        }
        numeric_features = sum(1 for t in feature_types.values() if t == "numeric")
        categorical_features = sum(1 for t in feature_types.values() if t == "categorical")

        outliers_map = (analysis_result or {}).get("outliers", {}).get("outliers", {}) or {}
        has_outliers = any(info.get("count", 0) > 0 for info in outliers_map.values())

        target_balance = (analysis_result or {}).get("target_balance", {}) or {}
        is_imbalanced = (not is_regression) and (not target_balance.get("is_balanced", True))

        if rows < 1000:
            size_category = "small"
        elif rows <= 10000:
            size_category = "medium"
        else:
            size_category = "large"

        scores = self._score_candidate_models(
            rows=rows,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            has_outliers=has_outliers,
            is_imbalanced=is_imbalanced,
            size_category=size_category,
            is_regression=is_regression,
        )

        ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)

        reasons = self._get_model_reasons(
            model_names=[name for name, _ in ranked],
            rows=rows,
            columns=profile.get("columns", df.shape[1]),
            problem_type=problem_type,
            size_category=size_category,
            has_outliers=has_outliers,
            is_imbalanced=is_imbalanced,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        min_recommended = min(3, len(ranked))
        recommended = []
        optional = []
        for i, (name, info) in enumerate(ranked):
            entry = {
                "name": name,
                "score": info["score"],
                "reason": reasons.get(name) or MODEL_REASON_FALLBACKS.get(name, "Included based on your dataset's characteristics."),
            }
            if i < min_recommended:
                entry["tag"] = "Recommended"
                entry["pre_selected"] = True
                recommended.append(entry)
            else:
                entry["tag"] = "Optional"
                entry["pre_selected"] = False
                optional.append(entry)

        return {
            "recommended": recommended,
            "optional": optional,
            "data_summary": {
                "rows": rows,
                "size_category": size_category,
                "has_outliers": has_outliers,
                "is_imbalanced": is_imbalanced,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            },
        }

    # ------------------------------------------------------------------
    def generate_ml_code(self, preprocessing_log, selected_models, best_model_name, best_params, problem_type) -> str:
        module_name, class_name = MODEL_IMPORTS.get(best_model_name, ("sklearn.ensemble", "RandomForestClassifier"))
        params_str = ", ".join(f"{k}={v!r}" for k, v in (best_params or {}).items())
        log_comments = "\n".join(f"# - {entry}" for entry in preprocessing_log) or "# (no preprocessing steps recorded)"

        if problem_type == "regression":
            metrics_import = "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"
            metrics_code = (
                "mae = mean_absolute_error(y_test, predictions)\n"
                "rmse = mean_squared_error(y_test, predictions) ** 0.5\n"
                "r2 = r2_score(y_test, predictions)\n"
                'print(f"MAE: {mae:.4f}")\n'
                'print(f"RMSE: {rmse:.4f}")\n'
                'print(f"R2 Score: {r2:.4f}")\n'
            )
        else:
            metrics_import = "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score"
            metrics_code = (
                "accuracy = accuracy_score(y_test, predictions)\n"
                'f1 = f1_score(y_test, predictions, average="weighted")\n'
                'precision = precision_score(y_test, predictions, average="weighted")\n'
                'recall = recall_score(y_test, predictions, average="weighted")\n'
                'print(f"Accuracy: {accuracy:.4f}")\n'
                'print(f"F1 Score: {f1:.4f}")\n'
                'print(f"Precision: {precision:.4f}")\n'
                'print(f"Recall: {recall:.4f}")\n'
            )

        code = f'''# Auto-generated ZeroCode training script
# Best model: {best_model_name}
# Problem type: {problem_type}

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from {module_name} import {class_name}
{metrics_import}

# ------------------------------------------------------------------
# Load your dataset here
# df = pd.read_csv("your_dataset.csv")

# ------------------------------------------------------------------
# Preprocessing steps applied by ZeroCode:
{log_comments}

# NOTE: replace TARGET_COLUMN with your actual target column name.
TARGET_COLUMN = "target"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
if numeric_cols:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ------------------------------------------------------------------
# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Best model: {best_model_name}, tuned with RandomizedSearchCV
model = {class_name}({params_str})
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# Evaluation
predictions = model.predict(X_test)
{metrics_code}'''

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, "model_training.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        return filepath.replace("\\", "/")

    # ------------------------------------------------------------------
    def generate_shap_analysis(
        self, model, X_train, X_test, feature_names, problem_type, model_name
    ) -> dict:
        try:
            X_train_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train, columns=feature_names)
            X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
            X_test_df = X_test_df.reset_index(drop=True)

            is_kernel = model_name not in SHAP_TREE_MODELS and model_name not in SHAP_LINEAR_MODELS
            max_samples = 50 if is_kernel else 200
            X_eval = X_test_df.iloc[: min(max_samples, len(X_test_df))].reset_index(drop=True)

            # A) build explainer based on model type
            if model_name in SHAP_TREE_MODELS:
                explainer = shap.TreeExplainer(model)
            elif model_name in SHAP_LINEAR_MODELS:
                explainer = shap.LinearExplainer(model, X_train_df)
            else:
                background = shap.sample(X_train_df, min(50, len(X_train_df)))
                explainer = shap.KernelExplainer(model.predict, background)

            # B) calculate SHAP values, handling both new Explanation objects
            # and legacy list/array outputs across explainer types
            try:
                shap_output = explainer(X_eval)
            except Exception:
                shap_output = explainer.shap_values(X_eval)

            values = shap_output.values if hasattr(shap_output, "values") else shap_output

            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            elif isinstance(values, np.ndarray) and values.ndim == 3:
                values = values[:, :, 1]

            values = np.asarray(values)

            os.makedirs(CHARTS_FOLDER, exist_ok=True)
            abs_mean = np.abs(values).mean(axis=0)
            top_n = min(15, len(feature_names))
            top_idx = np.argsort(abs_mean)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_idx]
            top_importances = abs_mean[top_idx]

            charts = {}

            kaleido.start_sync_server(silence_warnings=True)
            try:
                # C) Chart 1 — Feature Importance Bar
                try:
                    fig1 = go.Figure(
                        go.Bar(
                            x=top_importances[::-1],
                            y=top_features[::-1],
                            orientation="h",
                            marker=dict(
                                color=top_importances[::-1],
                                colorscale=[[0, "#06b6d4"], [1, "#8b5cf6"]],
                            ),
                        )
                    )
                    fig1.update_layout(
                        title="SHAP Feature Importance",
                        xaxis_title="Mean |SHAP value|",
                        template="plotly_dark",
                    )
                    path1 = os.path.join(CHARTS_FOLDER, "shap_importance.png")
                    fig1.write_image(path1)
                    charts["shap_importance"] = "outputs/charts/shap_importance.png"
                except Exception:
                    pass

                # D) Chart 2 — SHAP Summary Dot Plot
                try:
                    rng = np.random.default_rng(42)
                    n_feats = len(top_idx)
                    fig2 = go.Figure()
                    ticktext = [None] * n_feats
                    for pos, idx in enumerate(top_idx):
                        y_base = n_feats - 1 - pos
                        ticktext[y_base] = feature_names[idx]
                        feat_vals = X_eval.iloc[:, idx].astype(float).values
                        shap_vals_feat = values[: len(X_eval), idx]
                        jitter = rng.uniform(-0.35, 0.35, size=len(shap_vals_feat))
                        fig2.add_trace(
                            go.Scatter(
                                x=shap_vals_feat,
                                y=y_base + jitter,
                                mode="markers",
                                marker=dict(
                                    color=feat_vals,
                                    colorscale="RdBu_r",
                                    size=6,
                                    showscale=(pos == 0),
                                    colorbar=dict(title="Feature value") if pos == 0 else None,
                                ),
                                showlegend=False,
                            )
                        )
                    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig2.update_yaxes(tickvals=list(range(n_feats)), ticktext=ticktext)
                    fig2.update_layout(
                        title="SHAP Summary Plot — Feature Impact",
                        xaxis_title="SHAP value",
                        template="plotly_dark",
                    )
                    path2 = os.path.join(CHARTS_FOLDER, "shap_summary.png")
                    fig2.write_image(path2)
                    charts["shap_summary"] = "outputs/charts/shap_summary.png"
                except Exception:
                    pass

                # E) Chart 3 — Waterfall for sample 1 (index 0)
                try:
                    sample_shap = values[0]
                    order3 = np.argsort(np.abs(sample_shap))[::-1][:top_n]
                    feats3 = [feature_names[i] for i in order3]
                    vals3 = sample_shap[order3]
                    colors3 = ["#10b981" if v >= 0 else "#ef4444" for v in vals3]
                    fig3 = go.Figure(
                        go.Bar(
                            x=vals3[::-1],
                            y=feats3[::-1],
                            orientation="h",
                            marker_color=colors3[::-1],
                        )
                    )
                    fig3.update_layout(
                        title="Why Did The Model Predict This?",
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        template="plotly_dark",
                    )
                    path3 = os.path.join(CHARTS_FOLDER, "shap_waterfall.png")
                    fig3.write_image(path3)
                    charts["shap_waterfall"] = "outputs/charts/shap_waterfall.png"
                except Exception:
                    pass
            finally:
                kaleido.stop_sync_server()

            # F) top 10 feature importance list
            top10_idx = np.argsort(abs_mean)[::-1][: min(10, len(feature_names))]
            feature_importance = [
                {"feature": feature_names[i], "importance": round(float(abs_mean[i]), 4), "rank": rank + 1}
                for rank, i in enumerate(top10_idx)
            ]
            top3_names = [f["feature"] for f in feature_importance[:3]]

            # G) LLM explanation, with hardcoded fallback
            try:
                prompt = (
                    "Given these top features by SHAP importance:\n"
                    f"{feature_importance}\n"
                    f"Model used: {model_name}\n"
                    f"Problem type: {problem_type}\n\n"
                    "Write 2 sentences in plain English explaining which features drive "
                    "predictions most and what that means for this dataset. Keep it simple and clear."
                )
                explanation = self.llm.ask(prompt)
                if not explanation or explanation.startswith("Error:"):
                    raise ValueError("LLM explanation failed")
            except Exception:
                explanation = (
                    f"The model's decisions are most influenced by {', '.join(top3_names)}. "
                    "Higher values push the prediction up while lower values push it down."
                )

            # H) final return
            return {
                "status": "completed",
                "charts": charts,
                "feature_importance": feature_importance,
                "explanation": explanation,
            }
        except Exception as e:
            return {"status": "shap_failed", "reason": str(e)}

    # ------------------------------------------------------------------
    @staticmethod
    def _peek_default_model_names(problem_type: str, rows: int) -> list:
        # Mirrors select_models()'s name choices without instantiating
        # anything — used only to inform preprocessing (outliers/scaling)
        # of which models will train, before use_class_weight is known.
        if problem_type in CLASSIFICATION_TYPES:
            names = ["Random Forest", "Logistic Regression", "Decision Tree"]
            names.append("XGBoost" if rows > 1000 and XGBClassifier is not None else "SVM")
        else:
            names = ["Random Forest Regressor", "Linear Regression", "Decision Tree Regressor"]
            names.append("XGBoost Regressor" if rows > 1000 and XGBRegressor is not None else "SVR")
        return names

    # ------------------------------------------------------------------
    def run_full_ml_pipeline(
        self, df: pd.DataFrame, target_col: str, problem_type: str,
        extra_models: list = None, selected_metrics: list = None,
        user_preprocessing_choices: dict = None,
    ) -> dict:
        model_names_for_preprocessing = extra_models or self._peek_default_model_names(problem_type, len(df))

        prep = self.preprocess(
            df, target_col, problem_type,
            user_choices=user_preprocessing_choices,
            selected_models=model_names_for_preprocessing,
        )

        selected_models = {}
        reasoning = {}

        if extra_models:
            # An explicit list (e.g. from Model Recommendations) is the
            # complete desired training set, not an addition to the
            # automatic defaults.
            for name in extra_models:
                model = self.get_model_from_name(name, problem_type, prep["use_class_weight"])
                if model is None:
                    continue
                selected_models[name] = model
                reasoning[name] = "User selected this model manually"

        if not selected_models:
            selection = self.select_models(problem_type, df, prep["use_class_weight"])
            selected_models = selection["selected_models"]
            reasoning = selection["reasoning"]

        is_classification = problem_type in CLASSIFICATION_TYPES
        core_metrics = CORE_CLASSIFICATION_METRICS if is_classification else CORE_REGRESSION_METRICS
        effective_metrics = core_metrics + [m for m in (selected_metrics or []) if m not in core_metrics]

        train_result = self.train_models(
            selected_models, prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"], problem_type,
            selected_metrics=selected_metrics,
        )
        results = train_result["results"]

        leaderboard_result = self.generate_leaderboard(results, problem_type)
        best_model_name = leaderboard_result["best_model_name"]

        if best_model_name is None:
            return {
                "preprocessing_log": prep["preprocessing_log"],
                "model_reasoning": reasoning,
                "leaderboard": [],
                "best_model_name": None,
                "best_params": {},
                "tuning_score": None,
                "saved_files": {},
                "ml_code_path": None,
                "selected_metrics": effective_metrics,
                "status": "failed",
            }

        best_model = results[best_model_name]["model"]

        tune_result = self.tune_best_model(
            best_model_name, best_model, prep["X_train"], prep["y_train"], problem_type
        )

        save_result = self.save_model(
            tune_result["tuned_model"], best_model_name, prep["feature_names"], prep["scaler"]
        )

        ml_code_path = self.generate_ml_code(
            prep["preprocessing_log"], selected_models, best_model_name, tune_result["best_params"], problem_type
        )

        shap_result = self.generate_shap_analysis(
            tune_result["tuned_model"],
            prep["X_train"],
            prep["X_test"],
            prep["feature_names"],
            problem_type,
            best_model_name,
        )

        return {
            "preprocessing_log": prep["preprocessing_log"],
            "model_reasoning": reasoning,
            "leaderboard": leaderboard_result["leaderboard"],
            "best_model_name": best_model_name,
            "best_params": tune_result["best_params"],
            "tuning_score": tune_result["tuning_score"],
            "saved_files": save_result,
            "shap_analysis": shap_result,
            "ml_code_path": ml_code_path,
            "selected_metrics": effective_metrics,
            "status": "completed",
        }
