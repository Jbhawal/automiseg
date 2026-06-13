"""
Learnable Test-Time Adaptors (LTAs) — Section 3.5 of AutoMiSeg.

Defines the full hyperparameter search space used in Bayesian Optimization.
Each LTA parameter controls one image transformation or prompt configuration,
applied separately for the grounding model input and the segmentation model input.

Search space mirrors Table 3 from the paper.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


class LearningTestTimeAdaptor:
    """
    Namespace for LTA parameter definitions.
    Actual optimization is done in modules/tta.py via Optuna.
    """

    # ------------------------------------------------------------------
    # Parameter definitions for Optuna (used in tta.py)
    # ------------------------------------------------------------------
    SEARCH_SPACE = {
        # ---- GROUNDING model inputs ----
        "grd_hsv_hue":   ("int",   0,   20),    # HSV hue shift
        "grd_hsv_sat":   ("int",   0,   30),    # HSV saturation shift
        "grd_hsv_val":   ("int",   0,   30),    # HSV brightness shift
        "grd_r_shift":   ("int",   0,   20),    # RGB red channel offset
        "grd_g_shift":   ("int",   0,   20),    # RGB green channel offset
        "grd_b_shift":   ("int",   0,   20),    # RGB blue channel offset
        "grd_clahe_clip":("float", 0.0, 4.0),   # CLAHE clip limit
        "grd_clahe_grid":("int",   1,   4),      # CLAHE grid tile count
        "grd_edge_str":  ("float", 0.0, 1.0),   # Unsharp masking strength
        # Categorical: which of the 10 LLM-generated grounding prompts to use
        "grd_prompt_id": ("categorical", list(range(10))),

        # ---- SEGMENTATION model inputs ----
        "seg_hsv_hue":   ("int",   0,   20),
        "seg_hsv_sat":   ("int",   0,   30),
        "seg_hsv_val":   ("int",   0,   30),
        "seg_r_shift":   ("int",   0,   20),
        "seg_g_shift":   ("int",   0,   20),
        "seg_b_shift":   ("int",   0,   20),
        "seg_clahe_clip":("float", 0.0, 4.0),
        "seg_clahe_grid":("int",   1,   4),
        "seg_edge_str":  ("float", 0.0, 1.0),

        # ---- Prompt boosting ----
        # 0 = disabled; 1-5 = number of K-Means centroids passed to SAM
        "bst_k_points":  ("int",   0,   5),
    }

    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Identity / no-op parameter set (no transformation applied)."""
        return {
            # Grounding
            "grd_hsv_hue":    0,
            "grd_hsv_sat":    0,
            "grd_hsv_val":    0,
            "grd_r_shift":    0,
            "grd_g_shift":    0,
            "grd_b_shift":    0,
            "grd_clahe_clip": 0.0,
            "grd_clahe_grid": 1,
            "grd_edge_str":   0.0,
            "grd_prompt_id":  0,
            # Segmentation
            "seg_hsv_hue":    0,
            "seg_hsv_sat":    0,
            "seg_hsv_val":    0,
            "seg_r_shift":    0,
            "seg_g_shift":    0,
            "seg_b_shift":    0,
            "seg_clahe_clip": 0.0,
            "seg_clahe_grid": 1,
            "seg_edge_str":   0.0,
            # Prompt boosting
            "bst_k_points":   3,    # default: 3 clusters as per paper
        }

    @staticmethod
    def suggest(trial) -> Dict[str, Any]:
        """
        Called inside Optuna objective function.
        Uses trial.suggest_* to sample from the search space.
        """
        import optuna
        params = {}
        space = LearningTestTimeAdaptor.SEARCH_SPACE

        for name, spec in space.items():
            if spec[0] == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec[0] == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif spec[0] == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])

        return params
