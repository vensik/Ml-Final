import random
import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error

import torch

def set_seed(seed: int) -> None:
    """ Set the random seed for reproducibility. """

    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # Torch is not installed, skip setting seed for torch

def get_timestamp() -> str:
    """ Return current time for unique results """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_metrics(task_type: str, y_true, y_pred, y_proba=None) -> dict:
    """ Compute evaluation metrics based on the task type. """
    
    if task_type == "classification":
        metrics = {"accuracy": accuracy_score(y_true, y_pred)}
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        return metrics

    if task_type == "regression":
        return {
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "mae": mean_absolute_error(y_true, y_pred),
        }
    raise ValueError(f"Unsupported task type: {task_type}. Supported types are 'classification' and 'regression'.")

def log_result(results: list, model_name: str, fold: int, metrics: dict) -> None:
    """ Add one row to list of results"""
    row = {"model": model_name, "fold": fold, **metrics}
    results.append(row)

def save_results(results: list, path: Path) -> pd.DataFrame:
    """ Save existing results to csv and return DataFrame for quick view """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return df

def get_best_model(results: pd.DataFrame, task_type: str) -> str:
    """Select best model for submission based on task type."""
    scores = results.groupby("model").mean(numeric_only=True)

    if task_type == "classification":
        return scores["accuracy"].idxmax()
    
    if task_type == "regression":
        return scores["rmse"].idxmin()

    raise ValueError(f"Unsupported task type: {task_type}.")

def make_submission(test_df, predictions, cfg, path:Path) -> pd.DataFrame:
    """ Create Kaggle submission.csv """
    submission = pd.DataFrame({
        cfg.general.ID: test_df[cfg.general.ID],
        cfg.general.TARGET: predictions.astype(int),
    })
    submission.to_csv(path, index=False)
    return submission
