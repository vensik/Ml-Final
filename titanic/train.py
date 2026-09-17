import numpy as np
from omegaconf import OmegaConf

from data import postprocessing
from utils import compute_metrics, log_result

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# == DNN Model ==

class MLP(nn.Module):
    """MLP for binary classification"""
    def __init__(self, input_dim, epochs=100, batch_size=32, learning_rate=0.001, device="cpu", dropout=0.2):
        super().__init__()

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32,1)
        )

    def forward(self, x):
        return self.mlp(x) 

    def fit(self, X_train, y_train):
        df = TensorDataset(X_train, y_train)

        loader = DataLoader(df, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate)

        self.train()

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:

                optimizer.zero_grad()

                logits = self(X_batch)

                loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        self.eval()

        with torch.no_grad():
            logits = self(X)
            y_proba = torch.sigmoid(logits).squeeze(1).numpy()

        y_pred = (y_proba >= 0.5).astype(int)

        return y_pred, y_proba


# == Train functions ==

MODEL_REGISTRY = {  # baseline instead of linreg/logreg for list to have same keys
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
    "catboost": {
        "classification": (CatBoostClassifier, "tree"),
        "regression": (CatBoostRegressor, "tree"),
    },  
    "lightgbm":{
       "classification": (LGBMClassifier, "tree"),
       "regression": (LGBMRegressor, "tree"), 
    },   
    "xgboost":{
       "classification": (XGBClassifier, "tree"),
       "regression": (XGBRegressor, "tree"), 
    },
    "mlp": {
        "classification": (MLP, "torch"),
    } 
}

def get_model(name:str, seed: int, task_type: str, cfg, input_dim=None):
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

    if model_family == "torch":
        if input_dim is None:
            raise ValueError("input_dim is required for DNN")
        
        model = model_cls(input_dim=input_dim, **params)
        return model, model_family
    
    try:
        model = model_cls(random_state=seed, **params)
    except TypeError:
        model = model_cls(**params)

    return model, model_family

def train_model(model_name: str, X_train, y_train, cfg):
    """Train a single model."""

    task_type = cfg.general.TASK
    model_family = MODEL_REGISTRY[model_name][task_type][1]

    if model_family == "torch": # Train logic for PyTorch models
        prep = postprocessing("linear")
        X_train = prep.fit_transform(X_train)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        model, _ = get_model(model_name, cfg.general.SEED, task_type, cfg, input_dim=X_train.shape[1])
        model.fit(X_train, y_train)

        return model, prep
    
    model, model_family = get_model(model_name, cfg.general.SEED, task_type, cfg)
    
    pipeline = Pipeline([
        ("postprocess", postprocessing(model_family)),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    return pipeline, None

def predict(model, prep, X):
    """Make predictions using trained model."""

    if isinstance(model, MLP):
        X = prep.transform(X)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32)
        return model.predict(X)

    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba

def run_cv(model_name:str, X, y, folds, cfg, results: list) -> np.ndarray:
    """ Run a CV for single model """
    task_type = cfg.general.TASK
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model, prep = train_model(model_name, X_train, y_train, cfg)
        y_pred, y_proba = predict(model, prep, X_val)

        oof_preds[val_idx] = y_pred
        metrics = compute_metrics(task_type, y_val, y_pred, y_proba)
        log_result(results, model_name, fold, metrics)

    return oof_preds
