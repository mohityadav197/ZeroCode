import pickle

import pandas as pd

from agents.framework_models import parse_csv
from agents.ml_agent import MLAgent
from agents.worker import WorkerNode
from runtime_context import output_root, safe_output_path
from pathlib import Path


class MLWorker(WorkerNode):
    stage = "ml"

    def execute(self, request):
        prep = self.predecessor(request, "preprocessing")
        profile = self.predecessor(request, "profiler") or self.predecessor(request, "eda")
        if not prep or not profile:
            raise ValueError("ML requires profiling and preprocessing results")
        # Never deserialize a client-supplied artifact. This exact file was created by our worker.
        artifact = safe_output_path(prep["artifact_path"])
        if artifact != Path(output_root).resolve() / "prepared.pkl":
            raise ValueError("Unexpected preprocessing artifact")
        with artifact.open("rb") as source:
            prepared = pickle.load(source)
        parsed = parse_csv(request.filepath)
        frame = pd.read_csv(parsed["filepath"])
        result = MLAgent().run_full_ml_pipeline(
            frame, target_col=request.target_col or profile["target_suggestion"]["suggested_target"],
            problem_type=request.problem_type or profile["problem_type"]["problem_type"],
            extra_models=request.extra_models, selected_metrics=request.selected_metrics,
            user_preprocessing_choices=request.user_preprocessing_choices, prepared_data=prepared,
        )
        if result["status"] != "completed":
            raise RuntimeError("All requested ML models failed to train")
        return result
