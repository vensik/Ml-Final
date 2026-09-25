from pathlib import Path
from config import config
from data import load_data, preprocessing, gen_features, get_folds, prep_fold
from train import run_cv, train_model, predict
from utils import set_seed, save_results, get_timestamp, get_best_model, make_submission

MODELS_TO_RUN = [model for model, on in config.to_run.items() if on]

def main():
    set_seed(config.general.SEED)
    timestamp = get_timestamp()

    train_df, test_df = load_data(config)

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    train_fe = gen_features(train_df)
    test_fe = gen_features(test_df)
    
    X, y = train_fe.drop(columns=[config.general.TARGET]), train_fe[config.general.TARGET]
    folds = get_folds(X, y, config)

    best_model_path = Path(config.paths.RESULTS_DIR) / "best_model.txt"

    if config.cv.ON:
        results = []
        results_path = Path(config.paths.RESULTS_DIR) / f"cv_results_{timestamp}.csv"

        for model_name in MODELS_TO_RUN:
            run_cv(model_name, X, y, folds, config, results)

        df = save_results(results, results_path)
        print(df.groupby("model").mean(numeric_only=True))
        print(f"\nSaved results to {results_path}")

        best_model = get_best_model(df, config.general.TASK)
        best_model_path.write_text(best_model)
        print(f"Best model is {best_model}")

    else:
        if not best_model_path.exists():
            raise FileNotFoundError("best_model.txt not found. Run CV first")
        best_model = best_model_path.read_text().strip()

        X, test_fe = prep_fold(X, test_fe)

        model, prep = train_model(best_model, X, y, config)
        preds, _ = predict(model, prep, test_fe)

        submission_path = Path(config.paths.RESULTS_DIR) / f"submission_{timestamp}.csv"
        submission = make_submission(test_df, preds, config, submission_path)
        submission.head()
        print(submission.shape)
        print(submission.columns)

if __name__ == "__main__":
    main()