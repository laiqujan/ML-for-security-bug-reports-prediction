import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import re

import warnings
warnings.filterwarnings("ignore")

def evaluate_text_models_with_nested_gridsearch(df: pd.DataFrame, text_col: str = "text", label_col: str = "label",outer_folds: int = 5,
                                                inner_folds: int = 5, random_state: int = 42, n_jobs: int = -1,):
    
    data = df[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].astype(str)

    X = data[text_col].values
    y = data[label_col].values

    class_labels = [0, 1]

    # Pipelines and parameter grids
    models = {
        # Logistic Regression
        "logreg": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=3000, random_state=random_state,class_weight="balanced"))
            ]),
            "param_grid": {
                # TF-IDF parameters
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
                "tfidf__max_df": [0.9, 1.0],

                # Logistic Regression parameters
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l1", "l2"],
                "clf__solver": ["liblinear", "saga"],
            },
        },

        # SVM
        "svm": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", SVC(class_weight="balanced"))
            ]),
            "param_grid": {
                # TF-IDF parameters
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
                "tfidf__max_df": [0.9, 1.0],

                # SVM parameters
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
                "clf__gamma": ["scale", "auto", 0.01, 0.1, 1, 10],
            },
        },

        # Random Forest
        "rf": {
            "pipeline": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", RandomForestClassifier(random_state=random_state,class_weight="balanced"))
            ]),
            "param_grid": {
                # TF-IDF parameters
                "tfidf__ngram_range": [(1, 1), (1, 2)],
                "tfidf__min_df": [1, 2],
                "tfidf__max_df": [0.9, 1.0],

                # Random Forest parameters
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
            },
        },
    }

    # Outer CV
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    per_model_fold_metrics = {name: [] for name in models}
    per_model_best_params = {name: [] for name in models}

    # Nested CV loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=inner_folds,shuffle=True,random_state=random_state + fold_idx)

        for model_name, spec in models.items():

            grid = GridSearchCV(estimator=spec["pipeline"],param_grid=spec["param_grid"],scoring="f1",cv=inner_cv,n_jobs=n_jobs,
                                refit=True,verbose=0,)

            # Inner CV tuning + refit on full outer-train
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            per_model_best_params[model_name].append(grid.best_params_)

            # Evaluate on outer-test
            y_pred = best_model.predict(X_test)

            prec, rec, f1, _ = precision_recall_fscore_support(y_test,y_pred,labels=class_labels,average=None,zero_division=0)

            acc = accuracy_score(y_test, y_pred)

            per_model_fold_metrics[model_name].append({"fold": fold_idx,"accuracy": acc,"precision_class0": prec[0],"recall_class0": rec[0],
                                                       "f1_class0": f1[0],"precision_class1": prec[1],"recall_class1": rec[1],
                                                       "f1_class1": f1[1],})

    # Aggregate results
    summary = {}
    for model_name, fold_metrics in per_model_fold_metrics.items():
        dfm = pd.DataFrame(fold_metrics)

        summary[model_name] = {
            "mean_accuracy": dfm["accuracy"].mean(),
            "mean_precision_class0": dfm["precision_class0"].mean(),
            "mean_recall_class0": dfm["recall_class0"].mean(),
            "mean_f1_class0": dfm["f1_class0"].mean(),
            "mean_precision_class1": dfm["precision_class1"].mean(),
            "mean_recall_class1": dfm["recall_class1"].mean(),
            "mean_f1_class1": dfm["f1_class1"].mean(),
            "fold_metrics": dfm,
            "best_params_per_fold": per_model_best_params[model_name],
        }

    return summary

def preprocess(issues):
    processed_issues = []
    for issue in issues:
        issue = re.sub(r'```.*?```', ' ', issue, flags=re.DOTALL)
        issue = re.sub(r'\n', ' ', issue)
        issue = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', issue)
        issue = re.sub(r'\d+', ' ', issue)
        issue = re.sub(r'[^a-zA-Z0-9?\s]', ' ', issue)
        issue = re.sub(r'\s+', ' ', issue)
        processed_issues.append(issue)

    return processed_issues


# Below we call above methods for experiments
folder = "data_for_use"
datasets = ["Camel", "Ambari", "Derby", "Wicket"]
all_rows = []

for ds in datasets:
    # Build file paths
    test_path = os.path.join(folder, f"Test_{ds}.csv")
    train_path = os.path.join(folder, f"Train_{ds}.csv")

    # Load train and test
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    # Combine and shuffle
    combined = pd.concat([test_df, train_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=44).reset_index(drop=True)

    # Select and rename columns
    df = combined[["combined_text", "security"]].copy()
    df = df.rename(columns={"combined_text": "text", "security": "label"})
    
    # Cleaning
    df['text'] =  preprocess(df['text'])

    # Main method for evaluation
    results = evaluate_text_models_with_nested_gridsearch(df)

    print(f"\n\n==================== DATASET: {ds} ====================")

    for model_name, res in results.items():
        print(f"\n=== {model_name.upper()} ===")
        print(f"Accuracy (mean over 5 folds): {res['mean_accuracy']:.4f}")
        print(f"Class 0: P={res['mean_precision_class0']:.4f} "
              f"R={res['mean_recall_class0']:.4f} "
              f"F1={res['mean_f1_class0']:.4f}")
        print(f"Class 1: P={res['mean_precision_class1']:.4f} "
              f"R={res['mean_recall_class1']:.4f} "
              f"F1={res['mean_f1_class1']:.4f}")

        # Summary
        all_rows.append({
            "dataset": ds,
            "model": model_name,
            "mean_accuracy": res["mean_accuracy"],
            "mean_precision_class0": res["mean_precision_class0"],
            "mean_recall_class0": res["mean_recall_class0"],
            "mean_f1_class0": res["mean_f1_class0"],
            "mean_precision_class1": res["mean_precision_class1"],
            "mean_recall_class1": res["mean_recall_class1"],
            "mean_f1_class1": res["mean_f1_class1"],
        })

#results_df = pd.DataFrame(all_rows)
