"""Stage 5 - fine-tune a transformer text encoder for multi-label classification.

Sized for a single RTX 3070 Ti (8 GB). The defaults are deliberately
conservative; the knobs that matter when you hit OOM, in the order you should
reach for them:

    1. --grad-accum 2        (halves the memory of an effective batch, costs
                              nothing but wall time)
    2. --max-len 192         (attention memory is quadratic in sequence length,
                              so this is the biggest single lever)
    3. --batch-size 16
    4. --model-name sentence-transformers/all-MiniLM-L6-v2   (22M params vs
                              DistilBERT's 66M; roughly 3x headroom)

Do not reach for gradient checkpointing first. It trades ~30% throughput for
memory you probably do not need at these sizes.

The point of this stage is NOT to beat the linear baseline by a lot. It is to
find out whether it beats it at all, and by how much, so you can decide whether
the extra serving complexity is worth carrying. If the answer is "2% macro AP",
the honest engineering call is to ship the TF-IDF model.

Usage:
    python -m src.train_encoder --epochs 3
    python -m src.train_encoder --model-name sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from pyspark.sql import functions as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import ARTIFACTS_DIR, FEATURES_DIR, RANDOM_SEED, get_spark


@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 256
    batch_size: int = 32
    grad_accum: int = 1
    epochs: int = 3
    lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    train_rows: int = 400_000
    eval_rows: int = 60_000
    amp: bool = True
    random_init: bool = False
    seed: int = RANDOM_SEED


class ReviewDataset(Dataset):
    """Tokenises lazily in the worker rather than up front.

    Pre-tokenising 400k reviews at max_len 256 would be ~400MB of int64 held
    resident for the whole run. Doing it per batch keeps host RAM flat and the
    tokenizer is not the bottleneck once num_workers > 0.
    """

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
        }


def make_collate(tokenizer):
    """Pad to the longest sequence in the batch, not to max_len.

    Most Yelp reviews are far shorter than 256 tokens. Padding per batch
    instead of globally cuts average sequence length roughly in half, which is
    a ~2x speedup on attention for free.
    """

    def collate(batch):
        labels = torch.tensor(np.stack([b["labels"] for b in batch]))
        enc = tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch],
             "attention_mask": [b["attention_mask"] for b in batch]},
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate


class MultiLabelEncoder(nn.Module):
    """Encoder + mean pooling + linear head.

    Mean pooling over the attention mask rather than the [CLS] token: for a
    model that was not pretrained with a sentence-level objective, the CLS
    representation is close to arbitrary until it has been fine-tuned into
    meaning something, and mean pooling converges faster on small budgets.
    """

    def __init__(self, model_name: str, n_labels: int, dropout: float = 0.1,
                 random_init: bool = False):
        super().__init__()
        if random_init:
            # The control for the gate. The fine-tuned model differs from the
            # TF-IDF baseline on three axes at once -- subword tokenisation with
            # no OOV, word order, and pretraining -- so a win cannot attribute
            # itself to any one of them. Same architecture, same tokeniser, same
            # head, no pretrained weights: if this lands near the lexical
            # ceiling the margin was transfer, i.e. a better prior on rare terms
            # than 20k TF-IDF weights can estimate. If it clears the ceiling on
            # its own, the architecture is extracting something bag-of-words
            # cannot represent.
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                                  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(h.dtype)            # [B, T, 1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.head(self.dropout(pooled))


def load_frames(cfg: TrainConfig):
    spark = get_spark("yelp-encoder-data")
    try:
        base = FEATURES_DIR / "review_labeled"

        def take(split: str, limit: int):
            df = spark.read.parquet(str(base)).filter(F.col("split") == split)
            total = df.count()
            if total > limit:
                df = df.sample(False, limit / total, seed=cfg.seed)
            pdf = df.select("text", "labels").toPandas()
            y = np.vstack(pdf["labels"].apply(lambda a: np.asarray(a, dtype=np.int8)).values)
            return pdf["text"].fillna("").tolist(), y

        tr = take("train", cfg.train_rows)
        te = take("test", cfg.eval_rows)
    finally:
        spark.stop()
    return tr, te


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, targets = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(**batch)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(labels.numpy())
    return np.vstack(scores), np.vstack(targets)


def macro_ap(y_true: np.ndarray, y_score: np.ndarray,
             vocab: list[str] | None = None) -> tuple[float, float, list[dict]]:
    """Macro AP, its prevalence floor, and the per-label breakdown.

    Returning only the scalar throws away the array every interesting question
    needs -- whether a margin concentrates in rare labels, which categories a
    model is actually failing. Persisting it means each run carries its own
    breakdown instead of needing a re-evaluation script later.
    """
    aps, prevs, per_label = [], [], []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ap, prev = float(average_precision_score(yt, y_score[:, j])), float(yt.mean())
        aps.append(ap)
        prevs.append(prev)
        per_label.append({
            "category": vocab[j] if vocab else str(j),
            "prevalence": prev,
            "average_precision": ap,
            "lift": ap / prev if prev > 0 else None,
        })
    return float(np.mean(aps)), float(np.mean(prevs)), per_label


def main() -> None:
    p = argparse.ArgumentParser()
    for field, value in asdict(TrainConfig()).items():
        flag = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action="store_true" if not value else "store_false",
                           dest=field, default=value)
        else:
            p.add_argument(flag, type=type(value), default=value, dest=field)
    p.add_argument("--eval-only", action="store_true",
                   help="score the saved checkpoint and write its per-label "
                        "breakdown; no training")
    args = p.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[encoder] device={device} model={cfg.model_name}")
    if device.type == "cuda":
        print(f"[encoder] gpu={torch.cuda.get_device_name(0)} "
              f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    vocab = json.loads((ARTIFACTS_DIR / "category_vocab.json").read_text())["categories"]
    (tr_text, ytr), (te_text, yte) = load_frames(cfg)
    print(f"[encoder] train={len(tr_text):,} eval={len(te_text):,} labels={len(vocab)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(
        ReviewDataset(tr_text, ytr, tokenizer, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
        num_workers=4, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    eval_loader = DataLoader(
        ReviewDataset(te_text, yte, tokenizer, cfg.max_len),
        batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model = MultiLabelEncoder(cfg.model_name, len(vocab), random_init=cfg.random_init).to(device)

    if args.eval_only:
        # Score an existing checkpoint and write its per-label breakdown. Needed
        # because runs predating the per-label change persisted only the scalar,
        # and re-training to recover an array the forward pass already computes
        # would be absurd.
        ckpt = ARTIFACTS_DIR / "encoder" / "model.pt"
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        y_score, y_true = evaluate(model, eval_loader, device, cfg.amp)
        ap, floor, per_label = macro_ap(y_true, y_score, vocab)
        print(f"[encoder] eval-only  macro AP={ap:.4f}  floor={floor:.4f}  "
              f"lift={ap / floor:.2f}x  rows={y_true.shape[0]:,}")
        out = ARTIFACTS_DIR / "encoder_eval.json"
        out.write_text(json.dumps({
            "macro_ap": ap, "prevalence_floor": floor,
            "eval_rows": int(y_true.shape[0]),
            "model_name": cfg.model_name, "random_init": cfg.random_init,
            "per_label": per_label,
        }, indent=2))
        print(f"[encoder] per-label breakdown -> {out}")
        return

    # Rare labels get up-weighted, but the weight is capped. Uncapped
    # pos_weight on a label at 0.2% prevalence is 500x, which makes the loss
    # for that one label dominate the gradient and destabilises the encoder.
    pos = ytr.sum(axis=0).astype(np.float64)
    neg = len(ytr) - pos
    pos_weight = np.clip(np.divide(neg, np.maximum(pos, 1.0)), 1.0, 20.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )

    # Discriminative learning rates: the pretrained encoder needs small steps
    # so it is not wrecked, the randomly initialised head needs large ones.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg.lr},
            {"params": model.head.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_ap, history = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.amp and device.type == "cuda"):
                loss = criterion(model(**batch), labels) / cfg.grad_accum

            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum
            seen += 1

            if step % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 200 == 0:
                mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f} peak_vram={mem:.2f}GB")

        y_score, y_true = evaluate(model, eval_loader, device, cfg.amp)
        ap, floor, per_label = macro_ap(y_true, y_score, vocab)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "macro_ap": ap, "prevalence_floor": floor,
                        "lift": ap / floor if floor else None,
                        "eval_rows": int(y_true.shape[0]),
                        "seconds": time.time() - t0,
                        "per_label": per_label})
        print(f"[encoder] epoch {epoch}  loss={running / max(seen, 1):.4f}  "
              f"macro AP={ap:.4f}  floor={floor:.4f}  lift={ap / floor:.2f}x  "
              f"({time.time() - t0:.0f}s)")

        if ap > best_ap:
            best_ap = ap
            out = ARTIFACTS_DIR / "encoder"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model.pt")
            tokenizer.save_pretrained(out)
            (out / "train_config.json").write_text(
                json.dumps({**asdict(cfg), "n_labels": len(vocab), "best_macro_ap": ap}, indent=2)
            )
            print(f"[encoder] checkpoint saved (macro AP {ap:.4f}) -> {out}")

    (ARTIFACTS_DIR / "encoder_history.json").write_text(json.dumps(history, indent=2))
    print(f"[encoder] best macro AP {best_ap:.4f}")
    print("[encoder] compare against artifacts/baseline_metrics.json before "
          "committing to serving a transformer.")


if __name__ == "__main__":
    main()
