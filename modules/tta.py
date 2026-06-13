"""
Test-Time Adaptation — Section 3.7 of AutoMiSeg.

Uses Optuna with TPE (Tree-structured Parzen Estimator) to maximize
the proxy validation score over the LTA parameter space.

Key design decisions matching the paper:
  - Batch BO: optimize on a *subset* of test samples (up to Ns=100),
    then apply found config to the full test set.
  - Per-sample BO is explicitly discouraged (Figure 6 in paper shows overfitting).
  - Nt = 100 trials in the paper; we default to 50 for 4GB GPU.

Install:
    pip install optuna
"""

import optuna
import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image

from modules.lta import LearningTestTimeAdaptor

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TestTimeAdaptation:
    def __init__(self, pipeline, task, n_trials: int = 50):
        """
        pipeline : AutoMiSeg instance
        task     : TaskDefinition
        n_trials : number of Bayesian Optimization trials
        """
        self.pipeline = pipeline
        self.task = task
        self.n_trials = n_trials

    def optimize(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Run BO over LTA params on `images` (a subset of the test set).
        Returns the best LTA parameter dict.
        """
        print(f"[TTA] Starting Bayesian Optimization: {self.n_trials} trials, {len(images)} images.")

        def objective(trial):
            params = LearningTestTimeAdaptor.suggest(trial)
            scores = []
            for image in images:
                try:
                    mask = self.pipeline.predict(image, self.task, lta_params=params)
                    score = self.pipeline.validate(image, mask, self.task)
                    scores.append(score)
                except Exception as e:
                    # If a transform config causes a crash, penalize it
                    scores.append(0.0)
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_val = study.best_value
        print(f"[TTA] Best validation score: {best_val:.4f}")
        print(f"[TTA] Best LTA params: {best_params}")

        return best_params

    @staticmethod
    def load_params(path: str) -> Dict[str, Any]:
        """Load previously saved LTA params from JSON."""
        import json
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_params(params: Dict[str, Any], path: str):
        """Persist LTA params to JSON for reuse."""
        import json
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"[TTA] Params saved → {path}")
