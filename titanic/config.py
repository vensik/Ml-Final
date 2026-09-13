from omegaconf import OmegaConf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


config = OmegaConf.create({

    "general": {
        "SEED": 67,
        "EXPT_NAME": "titanic",
        "TASK": "classification"
    },

    "paths": {
        "DATA_DIR": str(DATA_DIR),
        "TRAIN_PATH": str(DATA_DIR / "train.csv"),
        "TEST_PATH": str(DATA_DIR / "test.csv"),
        "RESULTS_DIR": str(ROOT_DIR / "results")
    },
    "training": {
        "n_estimators": 200,
        "learning_rate": 0.01,
    },
    
    "cv": {
        "n_splits": 5,
        "stratified": True,
    },

    "models": {
        "baseline": {
            "classification": {"max_iter": 1000},
            "regression": {}
        },
        "knn": {"n_neighbors": 11},
        "tree": {"max_depth": 3},
        "rf": {"n_estimators": "${training.n_estimators}", "max_depth": 5},
        "catboost": {
            "n_estimators": "${training.n_estimators}",
            "learning_rate": "${training.learning_rate}",
            "depth": 4,
            "verbose": False,   
        },
        "lightgbm": {
            "num_leaves": 63,
            "n_estimators": 100,
            "learning_rate": "${training.learning_rate}",
            "max_depth": 10,
            "verbose": -1,
        },        
        "xgboost": {
            "n_estimators": "${training.n_estimators}",
            "learning_rate": "${training.learning_rate}",
            "max_depth": 3,
        },
    },
})

