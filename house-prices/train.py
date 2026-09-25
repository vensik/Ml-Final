import numpy as np
from omegaconf import OmegaConf

from data import postprocessing, prep_fold
from utils import compute_metrics, log_result

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# == DNN Model ==
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            self.best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class MLP(nn.Module):
    """MLP for binary classification"""
    def __init__(
            self, input_dim, device="cpu", task_type="classification", epochs=100,
            batch_size=32, learning_rate=0.001, dropout=0.0, patience=30, min_delta=0.0):
        super().__init__()

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.task_type = task_type
        self.patience = patience
        self.min_delta = min_delta

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32,1)
        )
        self.to(self.device)

    def forward(self, x):
        return self.mlp(x) 

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        df = TensorDataset(X_train, y_train)

        loader = DataLoader(df, batch_size=self.batch_size, shuffle=True)
        
        if self.task_type == "classification":
            criterion = nn.BCEWithLogitsLoss()
        elif self.task_type == "regression":
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)


        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(loader)

            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_val_device = X_val.to(self.device)
                    y_val_device = y_val.to(self.device)

                    output = self(X_val_device)
                    val_loss = criterion(output, y_val_device).item()

                scheduler.step(val_loss)
                stop = early_stopping(val_loss, self)

                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch + 1} / {self.epochs}, " 
                        f"train_loss={train_loss:.6f}, "
                        f"val_loss={val_loss:.6f}, "
                        f"lr={current_lr:.6f}"
                    )
                if stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1} / {self.epochs}, loss={train_loss:.6f}")

        early_stopping.restore_best(self)

        return self

    def predict(self, X):
        self.eval()
        X = X.to(self.device)

        with torch.no_grad():
            output = self(X)

        if self.task_type == "classification":
            y_proba = torch.sigmoid(output).squeeze(1)
            y_pred = (y_proba >= 0.5).int()

            return y_pred.cpu().numpy(), y_proba.cpu().numpy()
        
        if self.task_type == "regression":
            y_pred = output.squeeze(1)
            return y_pred.cpu().numpy(), None

        raise ValueError(f"Unsupported task type: {self.task_type}")


# == Train functions ==

MODEL_REGISTRY = {  # baseline instead of linreg/logreg for list to have same keys
    "baseline": {
        "classification": (LogisticRegression, "linear"),
        "regression": (Ridge, "linear")
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
        "regression": (MLP, "torch")
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
        
        model = model_cls(input_dim=input_dim, device=cfg.general.DEVICE, task_type=task_type, **params)
        return model, model_family
    
    try:
        model = model_cls(random_state=seed, **params)
    except TypeError:
        model = model_cls(**params)

    return model, model_family

def train_model(model_name: str, X_train, y_train, cfg, X_val=None, y_val=None):
    """Train a single model."""

    task_type = cfg.general.TASK
    model_family = MODEL_REGISTRY[model_name][task_type][1]

    if model_family == "torch": # Train logic for PyTorch models
        prep = postprocessing("linear", X_train)
        X_train = prep.fit_transform(X_train)
        if X_val is not None:
            X_val = prep.transform(X_val)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_val, "toarray"):
            X_val = X_val.toarray()

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

        if task_type == "regression":
            y_train = np.log1p(y_train.values)
            y_val = np.log1p(y_val.values)
        else:
            y_train = y_train.values
            y_val = y_val.values

        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        model, _ = get_model(model_name, cfg.general.SEED, task_type, cfg, input_dim=X_train.shape[1])
        model.fit(X_train, y_train, X_val, y_val)

        return model, prep
    
    model, model_family = get_model(model_name, cfg.general.SEED, task_type, cfg)
    
    pipeline = Pipeline([
        ("postprocess", postprocessing(model_family, X_train)),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    return pipeline, None

def predict(model, prep, X):
    """Make predictions using train model."""
    if isinstance(model, MLP):
        X = prep.transform(X)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32)

        y_pred, y_proba = model.predict(X)
        
        if model.task_type == "regression":
            y_pred = np.expm1(y_pred)
        
        return y_pred, y_proba

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
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        
        X_train, X_val = prep_fold(X_train, X_val)

        model, prep = train_model(model_name, X_train, y_train, cfg, X_val, y_val)
        y_pred, y_proba = predict(model, prep, X_val)

        oof_preds[val_idx] = y_pred
        metrics = compute_metrics(task_type, y_val, y_pred, y_proba)
        log_result(results, model_name, fold, metrics)

    return oof_preds

def tune_hyperparams(model_name, X, y, folds, cfg, params_grid, grid_mode=True):
    """"""
    task_type = cfg.general.TASK

    model_cls, model_family = MODEL_REGISTRY[model_name][task_type]

    extra_params = {}
    if model_name == "catboost":
        extra_params = {"verbose": False}
    elif model_name == "lightgbm":
        extra_params = {"verbose": -1, "n_jobs": 1} 
    elif model_name == "xgboost":
        extra_params = {"n_jobs": 1}

    try:
        base_model = model_cls(random_state=cfg.general.SEED, **extra_params)
    except TypeError:
        base_model = model_cls(**extra_params)

    pipeline = Pipeline([
        ("postprocess", postprocessing(model_family, X)),
        ("model", base_model),
    ])

    scoring = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"

    if grid_mode:
        search = GridSearchCV(pipeline, param_grid=params_grid, cv=folds, scoring=scoring, n_jobs=1)
    else:
        search = RandomizedSearchCV(
            pipeline, param_distributions=params_grid, n_iter=20, cv=folds,
            scoring=scoring, random_state=cfg.general.SEED, n_jobs=1
        )
    search.fit(X, y)

    return search

