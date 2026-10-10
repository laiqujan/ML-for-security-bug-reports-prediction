import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# data loading
folder = "data_for_use"
test_df = pd.read_csv(os.path.join(folder, "Test_Wicket.csv"))
train_df = pd.read_csv(os.path.join(folder, "Train_Wicket.csv"))

combined = pd.concat([test_df, train_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

df = combined[["combined_text", "security"]].copy()
df = df.rename(columns={"combined_text": "text", "security": "label"})

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

texts = df["text"].values
labels = df["label"].astype(int).values

# Config
MODEL_NAME = "roberta-base" #"bert-base-uncased"
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
N_SPLITS = 5
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

    sampler = WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)
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

# Training
def train_one_fold(train_idx, val_idx, fold):
    print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")

    X_train, y_train = texts[train_idx], labels[train_idx]
    X_val,   y_val   = texts[val_idx],   labels[val_idx]

    # Class weights for loss
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    train_ds = make_dataset(X_train, y_train)
    val_ds   = make_dataset(X_val,   y_val)

    train_loader = make_train_loader_with_sampler(train_ds, y_train, BATCH_SIZE)
    val_loader   = make_eval_loader(val_ds, BATCH_SIZE)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    optimizer = build_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #optimizer = AdamW(model.parameters(),lr=3e-5,  weight_decay=0.001)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
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

        # Evaluate each epoch
        precision, recall, f1, support, acc = evaluate(model, val_loader)
        print(
            f"Fold {fold+1}, Epoch {epoch+1}, TrainLoss={avg_loss:.4f}, ValAcc={acc:.4f}, "
            f"ValF1(minority)= {f1[1]:.4f}"
        )

    # Final Validation
  
    precision, recall, f1, support, acc = evaluate(model, val_loader)

    print(f"\nFold {fold+1} Results:")
    print(f"Accuracy: {acc:.4f}")
    for cls in [0, 1]:
        print(
            f"Class {cls}: Precision={precision[cls]:.4f}, "
            f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, Support={support[cls]}"
        )

    return precision, recall, f1, acc

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

all_prec, all_rec, all_f1, all_acc = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    p, r, f, acc = train_one_fold(train_idx, val_idx, fold)
    all_prec.append(p)
    all_rec.append(r)
    all_f1.append(f)
    all_acc.append(acc)

all_prec = np.array(all_prec)
all_rec  = np.array(all_rec)
all_f1   = np.array(all_f1)
all_acc  = np.array(all_acc)

avg_prec = all_prec.mean(axis=0)
avg_rec  = all_rec.mean(axis=0)
avg_f1   = all_f1.mean(axis=0)
avg_acc  = all_acc.mean()

print("\n========== FINAL AVERAGE RESULTS OVER 5 FOLDS ==========")
print(f"Accuracy: {avg_acc:.4f}")
for cls in [0, 1]:
    print(
        f"Class {cls}: Precision={avg_prec[cls]:.4f}, "
        f"Recall={avg_rec[cls]:.4f}, F1={avg_f1[cls]:.4f}"
    )
