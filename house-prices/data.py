from pathlib import Path
import pandas as pd
from config import config

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CAT_FEAT = 0
NUM_FEAT = 0
ALL_FEAT = 0
target = config.general.TARGET



def load_data(cfg):
    """ Load train/test data from config paths. """
    train_df = pd.read_csv(Path(cfg.paths.TRAIN_PATH))
    test_df = pd.read_csv(Path(cfg.paths.TEST_PATH))
    return train_df, test_df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """ Prepare existing features. """
    df = df.copy()
    

    return df

def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate new features. """
    df = df.copy()


    keep_cols = ALL_FEAT + ([target] if target in df.columns else [])
    return df[keep_cols]

def postprocessing(model_family: str) -> ColumnTransformer:
    """ Prepare DataFrame for model specifics """
    if model_family == "linear":
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEAT),
            ("num", StandardScaler(), NUM_FEAT),
        ])
    if model_family == "tree":
        return ColumnTransformer([
        ("passthrough", "passthrough", ALL_FEAT),
        ])
    raise ValueError(f"Unknown model family: {model_family}.")

def get_folds(X, y, cfg):
    """ Get cross-validation splits based on the config. """
    if cfg.cv.n_splits == 1:
        stratify = y if cfg.cv.stratified else None
        train_idx, val_idx = train_test_split(
            X.index, test_size=0.2, random_state=cfg.general.SEED, stratify=stratify
        )
        return [(X.index.get_indexer(train_idx), X.index.get_indexer(val_idx))]

    if cfg.cv.stratified:
        kf = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.general.SEED)
        return list(kf.split(X, y))
    kf = KFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.general.SEED)
    return list(kf.split(X))