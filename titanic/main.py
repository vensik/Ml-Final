from pathlib import Path
from config import config
from data import load_data, apply_fe, get_folds, ALL_FEAT, TARGET
from train import run_cv
from utils import set_seed, save_results
 
MODELS_TO_RUN = ["baseline", "knn", "tree", "rf"]

def main():
    set_seed(config.general.SEED)

    train_df, test_df = load_data(config)
    train_fe = apply_fe(train_df)

    X, y = train_fe[ALL_FEAT], train_fe[TARGET]
    folds = get_folds(X, y, config)

    results = []
    for model_name in MODELS_TO_RUN:
        run_cv(model_name, X, y, folds, config, results)

    results_path = Path(config.paths.RESULTS_DIR) / "cv_results.csv"
    df = save_results(results, results_path)

    print(df.groupby("model").mean(numeric_only=True))
    print(f"\nSaved results to {results_path}")

if __name__ == "__main__":
    main()