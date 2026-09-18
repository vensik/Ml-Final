from pathlib import Path
from config import config
from data import load_data, preprocessing, gen_features, get_folds, ALL_FEAT
from train import run_cv, train_model, predict
from utils import set_seed, save_results, get_timestamp, make_submission

MODELS_TO_RUN = [model for model, on in config.to_run.items() if on]

def main():
    set_seed(config.general.SEED)
    timestamp = get_timestamp()
    results = []
    
    train_df, test_df = load_data(config)

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    train_fe = gen_features(train_df)
    test_fe = gen_features(test_df)
    X, y = train_fe[ALL_FEAT], train_fe[config.general.TARGET]
    folds = get_folds(X, y, config)

    for model_name in MODELS_TO_RUN:
        run_cv(model_name, X, y, folds, config, results)

    results_path = Path(config.paths.RESULTS_DIR) / f"cv_results_{timestamp}.csv"
    df = save_results(results, results_path)

    print(df.groupby("model").mean(numeric_only=True))
    print(f"\nSaved results to {results_path}")

    BEST_MODEL = (df.groupby("model")["accuracy"].mean().idxmax()) 
    print(f"Best model is {BEST_MODEL}")
    model, prep = train_model(BEST_MODEL, X, y, config)
    preds, _ = predict(model, prep, test_fe[ALL_FEAT])
    submission_path = (Path(config.paths.RESULTS_DIR) / f"submission_{timestamp}.csv")

    submission = make_submission(test_df, preds, config, submission_path)
    submission.head()
    print(submission.shape)
    print(submission.columns)

if __name__ == "__main__":
    main()