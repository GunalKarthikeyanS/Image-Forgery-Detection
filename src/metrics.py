"""Evaluation: forged-region localization (Dice, IoU) and whole-image detection
(accuracy, F1, ROC-AUC). The original code measured Dice/IoU only, on forged images only."""
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def image_score(probs, top_frac=0.005):
    """How 'forged' an image looks: mean of its top 0.5% pixel probabilities."""
    flat = probs.flatten()
    k = max(1, int(top_frac * flat.numel()))
    return torch.topk(flat, k).values.mean().item()


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    dice, iou, scores, labels = [], [], [], []
    for x, y, is_forged in loader:
        out = model(x.to(device))
        if isinstance(out, (list, tuple)):
            out = out[-1]                                  # finest head
        probs = torch.sigmoid(out.float()).cpu()
        for p, t, f in zip(probs, y, is_forged):
            scores.append(image_score(p))
            labels.append(int(f))
            if f:                                          # localization: forged images only
                pred = (p > threshold).float()
                inter = (pred * t).sum().item()
                total = pred.sum().item() + t.sum().item()
                dice.append((2 * inter + 1e-6) / (total + 1e-6))
                iou.append((inter + 1e-6) / (total - inter + 1e-6))

    scores, labels = np.array(scores), np.array(labels)
    preds = (scores > threshold).astype(int)
    both = len(set(labels)) == 2
    return {
        "dice": float(np.mean(dice)) if dice else float("nan"),
        "iou": float(np.mean(iou)) if iou else float("nan"),
        "accuracy": float((preds == labels).mean()),
        "f1": float(f1_score(labels, preds)) if both else float("nan"),
        "auc": float(roc_auc_score(labels, scores)) if both else float("nan"),
        "n_images": int(len(labels)),
        "n_forged": int(labels.sum()),
    }
