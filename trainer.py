"""
Task 3 - Transfer learning on the scarce `organs` dataset.
Loads the source weights made by pretrain.py (orgs_pretrained.pt).

Regimes so far:
  zero-shot : load orgs weights, NO training -> how much transfers for free
"""
import json

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

from data import get_loaders
import models

CKPT        = "orgs_pretrained.pt"  
TARGET_DATA = "organs"              


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def new_model(in_ch, n_cls, device):
    # build a fresh ResNet18 of the right shape
    return models.ResNet18(in_channels=in_ch, num_classes=n_cls).to(device)


def test(model, test_loader, device):  # run the test images through the model and score the predictions
    model.eval()                       # eval mode -> BatchNorm uses stable running stats
    preds, labels = [], []
    with torch.no_grad():              # no gradients needed when only testing
        for x, y in test_loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()   # model's guess per image
            labels += y.tolist()                                    
    acc  = 100 * sum(p == t for p, t in zip(preds, labels)) / len(labels)
    prec = 100 * precision_score(labels, preds, average="macro", zero_division=0)
    rec  = 100 * recall_score(labels, preds, average="macro", zero_division=0)
    f1   = 100 * f1_score(labels, preds, average="macro", zero_division=0)
    return acc, prec, rec, f1


def main():
    with open("config.json") as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    tr, val, test_loader = get_loaders(TARGET_DATA, config["DATA_PATH"], config["BATCH_SIZE"])
    in_ch = tr.dataset.tensors[0].shape[1]
    n_cls = int(tr.dataset.tensors[1].max()) + 1
    print(f"target={TARGET_DATA}  in_channels={in_ch}  num_classes={n_cls}")

    results = {}

    # --- zero-shot: load orgs weights, test on organs with NO training ---
    set_seed()
    m = new_model(in_ch, n_cls, device)
    m.load_state_dict(torch.load(CKPT, map_location=device))   # pour in the orgs knowledge
    results["zero-shot"] = test(m, test_loader, device)

    # --- results table ---
    print("\n===== TASK 3 RESULTS (organs test set, target >= 40% acc) =====")
    print(f"{'regime':<12}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
    for name, (acc, prec, rec, f1) in results.items():
        print(f"{name:<12}{acc:>7.1f}%{prec:>7.1f}%{rec:>7.1f}%{f1:>7.1f}%")


if __name__ == "__main__":
    main()