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
    
    "cv": {
        "n_splits": 5,
        "stratified": True,
    },

    "models": {
        "baseline": {
            "classification": {"max_iter": 1000},
            "regression": {}
        },
        "knn": {"n_neighbors": 5},
        "tree": {"max_depth": None},
        "rf": {"n_estimators": 100, "max_depth": None},
    },
})

