import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# for loading data
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
df = df.sample(frac=1, random_state=222).reset_index(drop=True)

#print(df.shape)
#print(df["label"].value_counts())
#print(df["dataset"].value_counts())

# simple cleaning
def preprocess(issues):
    processed_issues = []
    for issue in issues:
        issue = str(issue)

        issue = re.sub(r'```.*?```', ' ', issue, flags=re.DOTALL)  
        issue = re.sub(r'\n', ' ', issue)                        
        issue = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            ' ',
            issue
        )
        issue = re.sub(r'\d+', ' ', issue)                       
        issue = re.sub(r'[^a-zA-Z0-9?\s]', ' ', issue)            
        issue = re.sub(r'\s+', ' ', issue).strip()               

        processed_issues.append(issue)
    return processed_issues

df["text"] = preprocess(df["text"])


# Config
MODEL_NAME = "roberta-base"
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def make_dataset(text_list, label_list):
    enc = tokenizer(
        list(text_list),
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(label_list, dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

def make_train_loader_with_sampler(dataset: TensorDataset, y_train: np.ndarray, batch_size: int):
    class_counts = np.bincount(y_train)
    class_counts = np.maximum(class_counts, 1)
    inv_class_freq = 1.0 / class_counts
    sample_weights = inv_class_freq[y_train]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def make_eval_loader(dataset: TensorDataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def build_optimizer(model, lr: float, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params, nodecay_params = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr
    )

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for input_ids, attention_mask, batch_labels in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        preds.extend(pred.tolist())
        trues.extend(batch_labels.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    precision, recall, f1, support = precision_recall_fscore_support(
        trues, preds, labels=[0, 1], zero_division=0
    )
    acc = accuracy_score(trues, preds)
    return precision, recall, f1, support, acc

# Training loop
def train_and_test_leave_one_dataset_out(train_df: pd.DataFrame, test_df: pd.DataFrame, test_dataset_name: str):
    X_train = train_df["text"].values
    y_train = train_df["label"].astype(int).values

    X_test = test_df["text"].values
    y_test = test_df["label"].astype(int).values

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Training split for test={test_dataset_name} has only one class. Cannot train.")
    if len(np.unique(y_test)) < 2:
        print(f"WARNING: Test split for {test_dataset_name} has only one class; metrics may be less informative.")

    # Class weights for loss
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    train_ds = make_dataset(X_train, y_train)
    test_ds  = make_dataset(X_test,  y_test)

    train_loader = make_train_loader_with_sampler(train_ds, y_train, BATCH_SIZE)
    test_loader  = make_eval_loader(test_ds, BATCH_SIZE)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    optimizer = build_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training
    print(f"\n==============================")
    print(f"TRAIN: {sorted(train_df['dataset'].unique().tolist())}")
    print(f"TEST : {test_dataset_name}")
    print(f"Train size={len(train_df)} | Test size={len(test_df)}")
    print(f"==============================")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Test={test_dataset_name} Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for input_ids, attention_mask, batch_labels in pbar:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / max(1, len(train_loader))

        #
        precision, recall, f1, support, acc = evaluate(model, test_loader)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | TrainLoss={avg_loss:.4f} | "
            f"TestAcc={acc:.4f} | TestF1(security=1)={f1[1]:.4f}"
        )

    # Final Test
    precision, recall, f1, support, acc = evaluate(model, test_loader)

    print(f"\nFINAL TEST RESULTS on {test_dataset_name}:")
    print(f"Accuracy: {acc:.4f}")
    for cls in [0, 1]:
        print(
            f"Class {cls}: Precision={precision[cls]:.4f}, "
            f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, Support={support[cls]}"
        )

    return {
        "test_dataset": test_dataset_name,
        "accuracy": float(acc),
        "non_security": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "security_bug": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }

# Run leave-one-dataset-out evaluation
results = []

for held_out in datasets:
    train_part = df[df["dataset"] != held_out].reset_index(drop=True)
    test_part  = df[df["dataset"] == held_out].reset_index(drop=True)

    res = train_and_test_leave_one_dataset_out(train_part, test_part, held_out)
    results.append(res)

print("\n==============================")
print("LEAVE-ONE-DATASET-OUT SUMMARY")
print("==============================")
for r in results:
    print(
        f"TEST={r['test_dataset']:7s} | "
        f"Acc={r['accuracy']:.4f} | "
        f"F1(sec=1)={r['security_bug']['f1']:.4f} | "
        f"Prec1={r['security_bug']['precision']:.4f} | "
        f"Rec1={r['security_bug']['recall']:.4f}"
    )
