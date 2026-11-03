import os
import pickle

import pandas as pd

from agents.framework_models import parse_csv, query_database, run_adk
from agents.ml_agent import MLAgent
from agents.worker import WorkerNode
from runtime_context import output_root


class PreprocessingAgent(WorkerNode):
    stage = "preprocessing"

    def execute(self, request):
        profile = self.predecessor(request, "profiler") or self.predecessor(request, "eda")
        if not profile:
            raise ValueError("Preprocessing requires a profiler or EDA result")
        target = request.target_col or profile["target_suggestion"]["suggested_target"]
        problem = request.problem_type or profile["problem_type"]["problem_type"]
        parsed = parse_csv(request.filepath)
        frame = pd.read_csv(parsed["filepath"])
        if target not in frame.columns or frame[target].isna().any():
            raise ValueError("Target must exist and contain no missing labels")
        query_database("SELECT session_id, updated_at FROM blackboard_state WHERE session_id = "
                       f"'{request.session_id}'")
        ml = MLAgent()
        selected = request.extra_models or ml._peek_default_model_names(problem, len(frame))
        prep = ml.preprocess(frame, target, problem, user_choices=request.user_preprocessing_choices, selected_models=selected)
        if not prep["feature_names"]:
            raise ValueError("Preprocessing removed all features; choose a dataset with usable predictors")
        path = os.path.join(output_root, "prepared.pkl")
        with open(path, "wb") as output:
            pickle.dump(prep, output)
        summary = run_adk("preprocessing", "Explain the actual preprocessing log in two sentences. "
                          "Do not suggest steps that were not performed.",
                          {"prompt": request.prompt, "preprocessing_log": prep["preprocessing_log"]}, [query_database])
        return {"artifact_path": path, "preprocessed_data_path": prep["preprocessed_data_path"],
                "preprocessing_code_path": prep["preprocessing_code_path"],
                "preprocessing_log": prep["preprocessing_log"], "feature_names": prep["feature_names"],
                "train_rows": len(prep["X_train"]), "test_rows": len(prep["X_test"]), "summary": summary}
