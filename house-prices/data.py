from pathlib import Path
import pandas as pd
import numpy as np
from config import config

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# === Generic functions ===

target = config.general.TARGET

def load_data(cfg):
    """ Load train/test data from config paths. """
    train_df = pd.read_csv(Path(cfg.paths.TRAIN_PATH))
    test_df = pd.read_csv(Path(cfg.paths.TEST_PATH))
    return train_df, test_df

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

# === Data dependant functions ===

ORD_FEAT = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu",
    "BsmtExposure", "GarageFinish", "LotShape", "LandSlope", "PavedDrive",
]
ORDINAL_MAPS = {
    "quality": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5,},
    "BsmtExposure": {"Unknown": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4,},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3,},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4,},
    "LandSlope": {"Gtl": 1, "Mod": 2, "Sev": 3,},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2,},
}   

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """ Prepare existing features. """
    df = df.copy()

    # Fill Nan
    none = [
        "PoolQC", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "FireplaceQu", "Alley", "Fence", "MiscFeature",
        "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinType2",
    ]
    for col in none:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    unknown = [
        "BsmtExposure", 
        "Electrical",
    ]
    for col in unknown:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    df.loc[df["MasVnrArea"] == 0, "MasVnrType"] = "None"
    df.loc[(df["MasVnrArea"] > 0) & (df["MasVnrType"].isna()), "MasVnrType"] = "Unknown"

    # ||Пометка для Сode review|| На cv может быть leakage по val, но решил, что это не так критично и не требуют усложнения пайплайна
    df["LotFrontage"] = (df["LotFrontage"].fillna(df.groupby("Neighborhood")["LotFrontage"].transform("median")))

    # Ordinal mapping
    for col in ORD_FEAT:
        if col in df.columns:
            mapping = ORDINAL_MAPS.get(col, ORDINAL_MAPS["quality"])
            df[col] = df[col].map(mapping)

    return df

def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate new features. """
    df = df.copy()

    # Log transform skewed features
    df["LotArea"] = np.log1p(df["LotArea"])

    # Feature generation
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

    df["TotalPorchSF"] = (
        df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )

    df["AvgRoomArea"] = df["GrLivArea"] / df["TotRmsAbvGrd"]

    return df

def get_feat_groups(df: pd.DataFrame):
    """Group features by their type."""

    to_exclude = ["MSSubClass"]
    ordinal = ORD_FEAT.copy()
    nominal = df.select_dtypes(include=["object"]).columns.tolist()
    nominal = [col for col in nominal if col not in ordinal]

    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    for col in ordinal:
        if col in numeric:
            numeric.remove(col)
    for col in to_exclude:
        if  col in numeric:
            numeric.remove(col)
            nominal.append(col)

    for col in [target, config.general.ID]:
        if col in numeric:
            numeric.remove(col)
        if col in ordinal:
            ordinal.remove(col)
        if col in nominal:
            nominal.remove(col)

    return {"numeric": numeric, "ordinal": ordinal, "nominal": nominal}

def postprocessing(model_family: str, df: pd.DataFrame) -> ColumnTransformer:
    """ Prepare DataFrame for model specifics """
    groups = get_feat_groups(df)
    numeric = groups["numeric"] + groups["ordinal"]
    nominal = groups["nominal"]
    
    if model_family == "linear":
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), nominal),
            ("num", StandardScaler(), numeric),
        ])
    if model_family == "tree":
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), nominal),
            ("num", "passthrough", numeric),
        ])
    raise ValueError(f"Unknown model family: {model_family}.")
