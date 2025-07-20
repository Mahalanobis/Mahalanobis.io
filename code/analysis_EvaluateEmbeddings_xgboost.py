import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import warnings
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import optuna

# Configurazioni
warnings.filterwarnings('ignore')
BGE_PATH = '/home/dario/Downloads/LLMFT4STATS/emotions_dataset_with_embeddings_BGE.parquet'
E5_PATH = '/home/dario/Downloads/LLMFT4STATS/emotions_dataset_with_embeddings_E5.parquet'
OPTUNA_TRIALS = 10  # Ridotto per debugging
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 5  # Ridotto per velocizzare

# Configurazione GPU - AGGIUNTA MANCANTE
GPU_CONFIG = {
    'tree_method': 'hist',
    'device': 'cuda',
    'predictor': 'gpu_predictor',
    'sampling_method': 'uniform'
}


def load_and_preprocess_data(file_path):
    """Carica e prepara i dati"""
    try:
        df = pd.read_parquet(file_path)
        le = LabelEncoder()
        y = le.fit_transform(df['Label'])
        X = np.vstack(df['embedding'].values)
        return X, y, len(le.classes_), df
    except Exception as e:
        print(f"Errore nel caricamento {file_path}: {str(e)}")
        return None, None, None, None


def objective(trial, X_train, y_train, num_classes):
    """Funzione obiettivo per Optuna"""
    params = {
        **GPU_CONFIG,  # Ora definito
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('lr', 0.05, 0.2),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.3),
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'random_state': RANDOM_STATE
    }

    # Usa solo 2 fold per velocizzare
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    acc_scores, f1_scores = [], []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = Pipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('xgb', XGBClassifier(**params))
        ])

        model.fit(
            X_tr, y_tr,
            xgb__eval_set=[(X_val, y_val)],
            xgb__verbose=0
        )

        y_pred = model.predict(X_val)
        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, average='macro'))

    trial.set_user_attr("accuracy", np.mean(acc_scores))
    return np.mean(f1_scores)


def evaluate_embeddings(X, y, num_classes, model_name):
    """Valutazione completa"""
    print(f"\n=== Valutazione {model_name} ===")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        lambda trial: objective(trial, X, y, num_classes),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True
    )

    best_params = {
        **study.best_params,
        **GPU_CONFIG,
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'random_state': RANDOM_STATE
    }

    # Validazione esterna
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    acc_scores, f1_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
        )

        model = Pipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('xgb', XGBClassifier(**best_params))
        ])

        model.fit(
            X_tr, y_tr,
            xgb__eval_set=[(X_val, y_val)],
            xgb__verbose=0
        )

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        acc_scores.append(acc)
        f1_scores.append(f1)
        print(f"Fold {fold + 1} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return {
        'accuracy': np.mean(acc_scores),
        'accuracy_std': np.std(acc_scores),
        'f1_macro': np.mean(f1_scores),
        'f1_macro_std': np.std(f1_scores),
        'best_params': best_params
    }


if __name__ == "__main__":
    results = {}
    for name, path in [('BGE', BGE_PATH), ('E5', E5_PATH)]:
        X, y, n_classes, df = load_and_preprocess_data(path)
        if X is not None:
            print(f"\nDataset: {name} | Samples: {len(X)} | Features: {X.shape[1]} | Classes: {n_classes}")
            res = evaluate_embeddings(X, y, n_classes, name)
            results[name] = res

    # Risultati finali
    print("\n=== RISULTATI FINALI ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"Accuracy: {metrics['accuracy']:.4f} (±{metrics['accuracy_std']:.4f})")
        print(f"F1-score: {metrics['f1_macro']:.4f} (±{metrics['f1_macro_std']:.4f})")
        print(f"Best params: {metrics['best_params']}")