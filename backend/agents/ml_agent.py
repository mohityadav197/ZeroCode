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
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    def preprocess(self, df: pd.DataFrame, target_col: str, problem_type: str) -> dict:
        log = []
        df = df.copy()
        rows = len(df)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # b) drop useless columns
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
            missing_pct = X[col].isna().mean() * 100
            if missing_pct > 50:
                X = X.drop(columns=[col])
                log.append(f"Dropped '{col}' — more than 50% missing ({missing_pct:.1f}%)")
                continue
            if missing_pct > 0:
                if pd.api.types.is_numeric_dtype(X[col]):
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    log.append(f"Filled '{col}' with median ({median_val})")
                else:
                    mode_series = X[col].mode()
                    mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    X[col] = X[col].fillna(mode_val)
                    log.append(f"Filled '{col}' with mode ('{mode_val}')")

        # d) encode categorical columns
        label_encoders = {}
        for col in list(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
                continue

            nunique = X[col].nunique()
            if nunique == 2:
                uniques = sorted(X[col].dropna().unique().tolist(), key=str)
                mapping = {uniques[0]: 0, uniques[1]: 1}
                X[col] = X[col].map(mapping)
                log.append(f"Binary encoded '{col}' ({mapping})")
            elif nunique < 10:
                dummies = pd.get_dummies(X[col], prefix=col).astype(int)
                X = X.drop(columns=[col])
                X = pd.concat([X, dummies], axis=1)
                log.append(f"One-Hot encoded '{col}' ({nunique} categories)")
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                log.append(f"Label encoded '{col}' ({nunique} categories)")

        # e) scale numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        if numeric_cols:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            log.append(f"Applied StandardScaler to numeric columns ({len(numeric_cols)} columns)")

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
                "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
                "AdaBoost": lambda: AdaBoostClassifier(),
                "Extra Trees": lambda: ExtraTreesClassifier(class_weight=class_weight, random_state=42),
                "Naive Bayes": lambda: GaussianNB(),
                "LightGBM": lambda: LGBMClassifier() if LGBMClassifier is not None else None,
                "CatBoost": lambda: CatBoostClassifier(verbose=0) if CatBoostClassifier is not None else None,
            }
        else:
            factories = {
                "KNN Regressor": lambda: KNeighborsRegressor(),
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
    def train_models(self, selected_models, X_train, X_test, y_train, y_test, problem_type: str) -> dict:
        results = {}

        for name, model in selected_models.items():
            try:
                start = time.time()
                model.fit(X_train, y_train)
                elapsed = time.time() - start
                preds = model.predict(X_test)

                if problem_type in CLASSIFICATION_TYPES:
                    results[name] = {
                        "model": model,
                        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                        "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
                        "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
                        "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
                        "training_time": f"{elapsed:.2f}s",
                    }
                else:
                    mae = mean_absolute_error(y_test, preds)
                    rmse = mean_squared_error(y_test, preds) ** 0.5
                    r2 = r2_score(y_test, preds)
                    results[name] = {
                        "model": model,
                        "mae": round(float(mae), 4),
                        "rmse": round(float(rmse), 4),
                        "r2": round(float(r2), 4),
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

        sorted_names = sorted(
            valid_results.keys(), key=lambda n: valid_results[n][metric_key], reverse=True
        )

        leaderboard = []
        for i, name in enumerate(sorted_names):
            r = valid_results[name]
            entry = {
                "rank": i + 1,
                "model": name,
                "training_time": r["training_time"],
                "is_best": i == 0,
            }
            if problem_type == "regression":
                entry.update({"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]})
            else:
                entry.update({
                    "accuracy": r["accuracy"],
                    "f1": r["f1"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                })
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
    def run_full_ml_pipeline(self, df: pd.DataFrame, target_col: str, problem_type: str, extra_models: list = None) -> dict:
        prep = self.preprocess(df, target_col, problem_type)

        selection = self.select_models(problem_type, df, prep["use_class_weight"])
        selected_models = selection["selected_models"]
        reasoning = selection["reasoning"]

        if extra_models:
            for name in extra_models:
                model = self.get_model_from_name(name, problem_type, prep["use_class_weight"])
                if model is None:
                    continue
                selected_models[name] = model
                reasoning[name] = "User selected this model manually"

        train_result = self.train_models(
            selected_models, prep["X_train"], prep["X_test"], prep["y_train"], prep["y_test"], problem_type
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
            "status": "completed",
        }
