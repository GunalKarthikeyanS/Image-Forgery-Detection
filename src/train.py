"""Training loop with validation-based checkpointing (best validation Dice)."""
import json
import time
from pathlib import Path

import torch

from metrics import evaluate


def train(model, train_loader, val_loader, loss_fn, device, out_dir,
          epochs=15, lr=5e-4, weight_decay=1e-4, amp=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, history = -1.0, []

    for epoch in range(1, epochs + 1):
        model.train()
        start, losses = time.time(), []
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x)
            out = [o.float() for o in out] if isinstance(out, list) else out.float()
            loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), **val,
               "minutes": (time.time() - start) / 60}
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val Dice {val['dice']:.3f}  "
              f"IoU {val['iou']:.3f}  acc {val['accuracy']:.3f}  AUC {val['auc']:.3f}  "
              f"({row['minutes']:.1f} min)")

        if val["dice"] > best:
            best = val["dice"]
            torch.save(model.state_dict(), out_dir / "best.pth")
        json.dump(history, open(out_dir / "history.json", "w"), indent=1)

    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
    return history
