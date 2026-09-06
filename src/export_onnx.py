"""Stage 6 - export the fine-tuned encoder to ONNX and prove the export is faithful.

The export itself is four lines. The part that matters is everything after it:
an ONNX file that loads is not the same as an ONNX file that computes the same
function. Operator fallbacks, opset differences, and fp16 casts all silently
change outputs, and the failure mode is a model that serves subtly wrong scores
for months without erroring once.

So this script exports, then re-runs both graphs over real review text and
fails loudly if they disagree beyond tolerance.

Usage:
    python -m src.export_onnx
    python -m src.export_onnx --opset 17 --tolerance 1e-4
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import ARTIFACTS_DIR
from src.train_encoder import MultiLabelEncoder

SAMPLE_TEXTS = [
    "Great little spot, the espresso was excellent and the staff remembered my order.",
    "Took my truck in for brakes and an alignment. Fair price, done same day.",
    "Waited 40 minutes for a table we had reserved, then the food came out cold.",
    "Clean rooms, easy parking, and the front desk sorted our late checkout without fuss.",
    "Short.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--tolerance", type=float, default=1e-4)
    args = p.parse_args()

    src = ARTIFACTS_DIR / "encoder"
    cfg = json.loads((src / "train_config.json").read_text())
    vocab = json.loads((ARTIFACTS_DIR / "category_vocab.json").read_text())["categories"]

    tokenizer = AutoTokenizer.from_pretrained(src)
    model = MultiLabelEncoder(cfg["model_name"], cfg["n_labels"])
    model.load_state_dict(torch.load(src / "model.pt", map_location="cpu"))
    model.eval()

    enc = tokenizer(SAMPLE_TEXTS, padding=True, truncation=True,
                    max_length=args.max_len, return_tensors="pt")
    dummy = (enc["input_ids"], enc["attention_mask"])

    onnx_path = ARTIFACTS_DIR / "encoder" / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # Both axes dynamic. Batch for throughput, sequence because we pad per
        # batch at serve time -- a graph frozen at 256 would force every
        # request to pad to 256 and waste most of the compute.
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # ---- parity check -----------------------------------------------------
    import onnxruntime as ort

    with torch.no_grad():
        torch_logits = model(enc["input_ids"], enc["attention_mask"]).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(
        ["logits"],
        {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()},
    )[0]

    max_abs = float(np.max(np.abs(torch_logits - onnx_logits)))
    print(f"[export] max |torch - onnx| = {max_abs:.3e}  (tolerance {args.tolerance:.0e})")
    if max_abs > args.tolerance:
        raise SystemExit(
            f"ONNX export diverges from PyTorch by {max_abs:.3e}. Do not serve this graph."
        )

    # ---- variable-length check -------------------------------------------
    # A graph exported with a fixed-length dummy batch can still have baked-in
    # shapes that only surface on a differently shaped input. Check explicitly.
    solo = tokenizer(SAMPLE_TEXTS[:1], padding=True, truncation=True,
                     max_length=args.max_len, return_tensors="pt")
    solo_onnx = sess.run(["logits"], {"input_ids": solo["input_ids"].numpy(),
                                      "attention_mask": solo["attention_mask"].numpy()})[0]
    if solo_onnx.shape != (1, cfg["n_labels"]):
        raise SystemExit(f"dynamic axes broken: got shape {solo_onnx.shape}")
    print(f"[export] dynamic shapes OK (batch=1 -> {solo_onnx.shape})")

    # ---- throughput reference --------------------------------------------
    warm = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    for _ in range(5):
        sess.run(["logits"], warm)
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(["logits"], warm)
    per_batch = (time.perf_counter() - t0) / 50
    print(f"[export] CPU EP: {per_batch * 1000:.2f} ms/batch of {len(SAMPLE_TEXTS)} "
          f"({per_batch / len(SAMPLE_TEXTS) * 1000:.2f} ms/review)")

    (ARTIFACTS_DIR / "encoder" / "serving_manifest.json").write_text(json.dumps({
        "onnx_path": str(onnx_path),
        "model_name": cfg["model_name"],
        "max_len": args.max_len,
        "opset": args.opset,
        "n_labels": cfg["n_labels"],
        "categories": vocab,
        "parity_max_abs_error": max_abs,
        "trained_macro_ap": cfg.get("best_macro_ap"),
    }, indent=2))
    print("[export] serving manifest written")


if __name__ == "__main__":
    main()
