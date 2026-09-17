from pathlib import Path
from config import config
from data import load_data, gen_features, get_folds, ALL_FEAT
from train import run_cv, train_model, predict
from utils import set_seed, save_results, get_timestamp, make_submission

MODELS_TO_RUN = [model for model, on in config.to_run.items() if on]

def main():
    set_seed(config.general.SEED)
    print("1. start")

    train_df, test_df = load_data(config)
    print("2. dataloader")
    train_fe = gen_features(train_df)
    test_fe = gen_features(test_df)
    print("3. features genereted")
    X, y = train_fe[ALL_FEAT], train_fe[config.general.TARGET]
    print("4. X/y preped")
    folds = get_folds(X, y, config)
    print("5.folds preped")

    timestamp = get_timestamp()
    results = []
    for model_name in MODELS_TO_RUN:
        print(f"6. training {model_name}")
        run_cv(model_name, X, y, folds, config, results)
        print(f"7. finished {model_name}")

    results_path = Path(config.paths.RESULTS_DIR) / f"cv_results_{timestamp}.csv"
    df = save_results(results, results_path)

    print(df.groupby("model").mean(numeric_only=True))
    print(f"\nSaved results to {results_path}")

    BEST_MODEL = (df.groupby("model")["accuracy"].mean().idxmax()) 
    model, prep = train_model(BEST_MODEL, X, y, config)
    preds, _ = predict(model, prep, test_fe[ALL_FEAT])
    submission_path = (Path(config.paths.RESULTS_DIR) / f"submission_{timestamp}.csv")

    submission = make_submission(test_df, preds, config, submission_path)
    submission.head()
    print(submission.shape)
    print(submission.columns)

if __name__ == "__main__":
    main()