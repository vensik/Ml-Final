from pathlib import Path
import pandas as pd
from config import config

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CAT_FEAT = ["Pclass", "Sex", "Embarked", "Initial", "Fare_cat"]
NUM_FEAT = ["SibSp", "Parch", "Age_range", "FamSize", "Alone"]
ALL_FEAT = CAT_FEAT + NUM_FEAT
target = config.general.TARGET

INITIAL_REPLACE = {
    "Mlle": "Miss", "Mme": "Miss", "Ms": "Miss",
    "Dr": "Mr", "Major": "Mr", "Capt": "Mr", "Sir": "Mr", "Don": "Mr",
    "Lady": "Mrs", "Countess": "Mrs", "Dona": "Mrs",
    "Jonkheer": "Other", "Col": "Other", "Rev": "Other",
}
AGE_BY_INITIAL = {"Mr": 33, "Mrs": 36, "Miss": 22, "Master": 5, "Other": 46}


def load_data(cfg):
    """ Load train/test data from config paths. """
    train_df = pd.read_csv(Path(cfg.paths.TRAIN_PATH))
    test_df = pd.read_csv(Path(cfg.paths.TEST_PATH))
    return train_df, test_df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """ Prepare existing features. """
    df = df.copy()
    
    # Extract Initials
    df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    df["Initial"] = df["Initial"].replace(INITIAL_REPLACE)

    # Fill missing Age values based on Initials
    for initial, age in AGE_BY_INITIAL.items():
        mask = df["Age"].isnull() & (df["Initial"] == initial)
        df.loc[mask, "Age"] = age
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Other missing values
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df

def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate new features. """
    df = df.copy()

    # Create Age ranges
    df["Age_range"] = 0
    df.loc[df["Age"] <= 16, "Age_range"] = 0
    df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age_range"] = 1
    df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age_range"] = 2
    df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age_range"] = 3
    df.loc[df["Age"] > 64, "Age_range"] = 4

    # FamSize and Alone
    df["FamSize"] = df["SibSp"] + df["Parch"]
    df["Alone"] = (df["FamSize"] == 0).astype(int)

    # Fare_cat 
    df["Fare_cat"] = 0
    df.loc[df["Fare"] <= 7.91, "Fare_cat"] = 0
    df.loc[(df["Fare"] > 7.91) & (df["Fare"] <= 14.454), "Fare_cat"] = 1
    df.loc[(df["Fare"] > 14.454) & (df["Fare"] <= 31), "Fare_cat"] = 2
    df.loc[df["Fare"] > 31, "Fare_cat"] = 3

    # Categorical features to numeric values
    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
    df['Initial'] = df['Initial'].replace({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4})

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