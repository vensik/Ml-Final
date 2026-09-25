from omegaconf import OmegaConf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


config = OmegaConf.create({

    "general": {
        "SEED": 67,
        "EXPT_NAME": "house-prices",
        "TASK": "regression",

        "TARGET": "SalePrice",
        "ID": "Id",

        "DEVICE": "cpu",
    },

    "paths": {
        "DATA_DIR": str(DATA_DIR),
        "TRAIN_PATH": str(DATA_DIR / "train.csv"),
        "TEST_PATH": str(DATA_DIR / "test.csv"),
        "RESULTS_DIR": str(ROOT_DIR / "results")
    },
    "training": {
        "n_estimators1": 200,
        "n_estimators2": 400,
        "learning_rate": 0.03,
    },
    
    "cv": {
        "ON": True,
        "n_splits": 5,
        "stratified": False,
    },

    "models": {
        "baseline": {
            "classification": {"max_iter": 1000},
            "regression": {}
        },
        "knn": {"n_neighbors": 11},
        "tree": {"max_depth": 3},
        "rf": {
            "n_estimators": "${training.n_estimators1}",
            "max_depth": 6,
            "min_samples_leaf": 5
        },
        "catboost": {
            "n_estimators": "${training.n_estimators2}",
            "learning_rate": "${training.learning_rate}",
            "depth": 5,
            "verbose": False,
            "thread_count": 1,   
        },
        "lightgbm": {
            "num_leaves": 63,
            "n_estimators": "${training.n_estimators2}",
            "learning_rate": "${training.learning_rate}",
            "max_depth": 11,
            "verbose": -1,
            "n_jobs": 1,
        },        
        "xgboost": {
            "n_estimators": "${training.n_estimators2}",
            "learning_rate": 0.03,
            "max_depth": 4,
            "n_jobs": 1,
        },
        "mlp": {
            "epochs": 300,
            "batch_size": 32,
            "learning_rate": 0.003,
            "dropout": 0.0,
        },
    },
    "to_run": {
        "baseline": True,
        "knn": False,
        "tree": False,
        "rf": False,
        "catboost": False,
        "lightgbm": False,
        "xgboost": False,
        "mlp": True,
    },
})

