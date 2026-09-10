import numpy as np
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from data import preprocessor
from utils import compute_metrics, log_result

"""baseline instead of linreg/logreg for list to have same keys"""
MODEL_REGISTRY = { 
    "baseline": {
        "classification": (LogisticRegression, "linear"),
        "regression": (LinearRegression, "linear")
    },
    "knn": {
        "classification": (KNeighborsClassifier, "linear"),
        "regression": (KNeighborsRegressor, "linear"),
    },
    "tree": {
        "classification": (DecisionTreeClassifier, "tree"),
        "regression": (DecisionTreeRegressor, "tree"),
    },
    "rf": {
        "classification": (RandomForestClassifier, "tree"),
        "regression": (RandomForestRegressor, "tree"),
    },
}

def get_model(name:str, seed: int, task_type: str, cfg):
    """Get model by name and task type from registy + params from cfg"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}, Availiable: {list(MODEL_REGISTRY)}")
    if task_type not in MODEL_REGISTRY[name]:
        raise ValueError(f"Model {name} has no {task_type} variant")

    model_cls, model_family = MODEL_REGISTRY[name][task_type]

    raw_cfg = cfg.models[name]
    if task_type in raw_cfg: # params depends by task
        params = OmegaConf.to_container(raw_cfg[task_type], resolve=True)
    else: # same params for both tasks
        params = OmegaConf.to_container(raw_cfg, resolve=True)

    try:
        model = model_cls(random_state=seed, **params)
    except TypeError:
        model = model_cls(**params)

    return model, model_family

def run_cv(model_name:str, X, y, folds, cfg, results: list) -> np.ndarray:
    """ Run a CV for single model """
    task_type = cfg.general.TASK
    model, model_family = get_model(model_name, cfg.general.SEED, task_type, cfg)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = Pipeline([
            ("preprocess", preprocessor(model_family)),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        oof_preds[val_idx] = y_pred

        y_proba = None
        if task_type == "classification":
            y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(task_type, y_val, y_pred, y_proba)
        log_result(results, model_name, fold, metrics)

    return oof_preds

def fit_predict(model_name: str, X, y, X_test, cfg):
    """"""
    model, model_family = get_model(model_name, cfg.general.SEED, cfg.general.TASK)
    pipeline = Pipeline([
        ("preprocess", preprocessor(model_family)),
        ("model", model),
    ])
    pipeline.fit(X,y)
    return pipeline.predict(X_test)