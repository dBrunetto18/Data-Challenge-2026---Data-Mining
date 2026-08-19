import polars as pl
import optuna
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# GLOBAL CONFIGURATION
BASE_PATH = Path("C:/University/DATA MINING/data challenge 2026")
RANDOM_STATE = 42
USE_AUGMENTED_DATA = False
SCORING = 'neg_log_loss'

N_TRIALS_RF = 50
N_TRIALS_XGB = 50
XGB_COMPLEXITY = True


def load_data(base_path: Path, use_augmented: bool):
    """Load train and test data based on your chosen configuration."""
    prefix = "augmented_" if use_augmented else ""
    train_file = "train_augmented.csv" if use_augmented else "training.csv"
    test_file = "test_augmented.csv" if use_augmented else "test.csv"
    
    data_train = pl.read_csv(base_path / train_file)
    data_test = pl.read_csv(base_path / test_file)
    
    X_train = data_train.drop(['contraceptive', 'ID']).to_pandas()
    y_train = data_train['contraceptive'].to_pandas()
    X_test = data_test.drop('ID').to_pandas()
    test_ids = data_test['ID']
    
    return X_train, y_train, X_test, test_ids

def get_preprocessor(use_augmented: bool):
    """Builds the ColumnTransformer dynamically."""
    edu_cats = ['None', 'Low', 'Intermediate', 'High']
    child_state = ['0', '1', 'More than 1']
    
    onehot_cols = ['religion', 'area']
    if not use_augmented:
        onehot_cols.insert(0, 'state') # 'state' only available for augmented case
        
    return ColumnTransformer(
        transformers=[
            ('OneHot', OneHotEncoder(handle_unknown='ignore'), onehot_cols),
            ('Ordinal', OrdinalEncoder(categories=[edu_cats, child_state]), ['education', 'child'])
        ],
        remainder='passthrough'
    )

# HYPERPARAMETERS
def rf_param_grid(trial):
    return {
        'clf__n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'clf__max_depth': trial.suggest_int('max_depth', 3, 20),
        'clf__min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }

def xgb_param_grid(trial, complexity: bool):
    params = {
        'clf__n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'clf__max_depth': trial.suggest_int('max_depth', 3, 20),
        'clf__learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True)
    }
    
    if complexity:
        params.update({
            'clf__subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'clf__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'clf__reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'clf__reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'clf__min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'clf__gamma': trial.suggest_float('gamma', 0.0, 5.0)
        })
    return params

def create_objective(pipeline, X, y, cv, param_grid_func):
    """Create a closure (internal function) for Optuna encapsulating the data."""
    def objective(trial):
        params = param_grid_func(trial)
        pipeline.set_params(**params)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        return scores.mean()
    return objective

# MAIN
def main():
    # Data Load
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data(BASE_PATH, USE_AUGMENTED_DATA)
    
    # Preprocessing and Pipeline
    preprocessor = get_preprocessor(USE_AUGMENTED_DATA)
    
    sKF = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    
    pipe_rf = Pipeline([
        ('Transformer', preprocessor), 
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE)) 
    ])
    
    pipe_xgb = Pipeline([
        ('Transformer', preprocessor), 
        ('clf', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')) 
    ])

    # Random Forest Optimization
    print("\nStarting Random Forest optimization...")
    study_rf = optuna.create_study(direction='maximize', study_name="Opt_RF")
    obj_rf = create_objective(pipe_rf, X_train, y_train, sKF, rf_param_grid)
    study_rf.optimize(obj_rf, n_trials=N_TRIALS_RF)

    # XGBoost Optimization
    print("\nStarting XGBoost optimization...")
    study_xgb = optuna.create_study(direction='maximize', study_name="Opt_XGB")
    obj_xgb = create_objective(pipe_xgb, X_train, y_train, sKF, 
                               lambda trial: xgb_param_grid(trial, XGB_COMPLEXITY))
    study_xgb.optimize(obj_xgb, n_trials=N_TRIALS_XGB)

    # Results
    print('\n---- RANDOM FOREST RESULTS ----')
    print("Best Score:", study_rf.best_value)
    print("Best parameters:", study_rf.best_params)
    
    print('\n---- XGBOOST RESULTS ----')
    print("Best Score:", study_xgb.best_value)
    print("Best parameters:", study_xgb.best_params)

    # Model selection
    if study_rf.best_value > study_xgb.best_value:
        print("\nBest model: Random Forest.")
        best_pipeline = pipe_rf
        best_params = study_rf.best_params
    else:
        print("\nBest model: XGBoost.")
        best_pipeline = pipe_xgb
        best_params = study_xgb.best_params

    # Final training
    print("Training the model on the whole dataset")
    formatted_params = {f'clf__{k}': v for k, v in best_params.items()}
    best_pipeline.set_params(**formatted_params)
    best_pipeline.fit(X_train, y_train)
    
    # Previsione e salvataggio
    proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    submission = pl.DataFrame({'ID': test_ids, 'prediction': proba})
    
    suffix = "_AUGMENTED.csv" if USE_AUGMENTED_DATA else ".csv"
    file_name = f"submission_{SCORING}{suffix}"
    path_complete = BASE_PATH / file_name
    
    submission.write_csv(path_complete)
    print(f"\nPrediction successfully saved at: {path_complete}")

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    main()