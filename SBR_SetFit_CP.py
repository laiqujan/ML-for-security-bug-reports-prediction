import os
import re
import random
import numpy as np
import pandas as pd
import torch

from setfit import SetFitModel
from setfit import SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb
wandb.init(mode="disabled")
import warnings
warnings.filterwarnings("ignore")

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

folder = "data_for_use"
datasets = ["Camel", "Ambari", "Derby", "Wicket"]

all_dfs = []
for ds in datasets:
    test_path = os.path.join(folder, f"Test_{ds}.csv")
    train_path = os.path.join(folder, f"Train_{ds}.csv")

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    combined = pd.concat([test_df, train_df], ignore_index=True)
    combined = combined[["combined_text", "security"]].copy()
    combined = combined.rename(columns={"combined_text": "text", "security": "label"})
    combined["dataset"] = ds
    all_dfs.append(combined)

df = pd.concat(all_dfs, ignore_index=True)
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

print("Total:", df.shape, "\nLabel counts:\n", df["label"].value_counts(), "\nDataset counts:\n", df["dataset"].value_counts())

# Preprocess
def preprocess(issues):
    processed = []
    for issue in issues:
        issue = str(issue)
        issue = re.sub(r'```.*?```', ' ', issue, flags=re.DOTALL)
        issue = re.sub(r'\n', ' ', issue)
        issue = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', issue)
        issue = re.sub(r'\d+', ' ', issue)
        issue = re.sub(r'[^a-zA-Z0-9?\s]', ' ', issue)
        issue = re.sub(r'\s+', ' ', issue).strip()
        processed.append(issue)
    return processed

df["text"] = preprocess(df["text"])
df["label"] = df["label"].astype(int)

# Leave-one-dataset-out runs
def eval_one_run(train_df_run: pd.DataFrame, test_df_run: pd.DataFrame, held_out: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)

    train_set = Dataset.from_pandas(train_df_run[["text", "label"]].reset_index(drop=True))
    test_set  = Dataset.from_pandas(test_df_run[["text", "label"]].reset_index(drop=True))

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_set,
        loss_class=CosineSimilarityLoss,
        num_iterations=5,
        num_epochs=1,
        batch_size=16,
    )
    trainer.train()

    y_pred = trainer.model.predict(test_set["text"])
    y_true = np.array(test_set["label"])

    res = {
        "test_dataset": held_out,
        "accuracy": float(accuracy_score(y_true, y_pred)),

        "sec_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sec_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sec_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),

        "nonsec_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "nonsec_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "nonsec_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),

        "train_size": int(len(train_df_run)),
        "test_size": int(len(test_df_run)),
    }
    return res

results = []
for held_out in datasets:
    train_part = df[df["dataset"] != held_out].copy()
    test_part  = df[df["dataset"] == held_out].copy()

    print(f"\n==============================")
    print(f"TRAIN on {sorted(train_part['dataset'].unique().tolist())}")
    print(f"TEST  on {held_out}  | train={len(train_part)} test={len(test_part)}")
    print(f"==============================")

    results.append(eval_one_run(train_part, test_part, held_out))

# Summary
print("\n========== LEAVE-ONE-DATASET-OUT SUMMARY ==========")
for r in results:
    print(
        f"TEST={r['test_dataset']:7s} | "
        f"Acc={r['accuracy']:.4f} | "
        f"F1(sec=1)={r['sec_f1']:.4f} | "
        f"P1={r['sec_precision']:.4f} R1={r['sec_recall']:.4f}"
    )

