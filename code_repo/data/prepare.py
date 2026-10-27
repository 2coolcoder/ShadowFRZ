"""Download and preprocess every dataset/checkpoint the study needs (run once).

  CIFAR-10/100      -> $SYSDL_CACHE/cifar/{cifar10,cifar100}.npz  (uint8 NHWC + labels)
  WikiText-103      -> $SYSDL_CACHE/wt103/{train,valid}_128.npy   (uint16 token ids, packed
                       [CLS] 126 tokens [SEP] sequences, bert-base-uncased WordPiece)
  GLUE SST-2, MNLI  -> HF datasets cache
  bert-base/large   -> HF hub cache (public checkpoints for the fine-tuning runs)
"""
import os
import sys

import numpy as np

CACHE = os.environ["SYSDL_CACHE"]
SEQ = 128


def cifar():
    # HF mirrors of the official CIFAR archives (the cs.toronto.edu host ran at ~60 kB/s here).
    from datasets import load_dataset
    out = os.path.join(CACHE, "cifar")
    os.makedirs(out, exist_ok=True)
    for name, repo, lab_col in (("cifar10", "uoft-cs/cifar10", "label"),
                                ("cifar100", "uoft-cs/cifar100", "fine_label")):
        path = os.path.join(out, f"{name}.npz")
        if os.path.exists(path):
            continue
        ds = load_dataset(repo)
        arr = {}
        for split, key in (("train", "tr"), ("test", "te")):
            arr["x" + key] = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[split]["img"]])
            arr["y" + key] = np.asarray(ds[split][lab_col], dtype=np.int16)
        np.savez(path, **arr)
        print(name, arr["xtr"].shape, arr["xte"].shape, "classes", int(arr["ytr"].max()) + 1)


def wikitext():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    out = os.path.join(CACHE, "wt103")
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    for split in ("train", "validation"):
        path = os.path.join(out, f"{'valid' if split == 'validation' else split}_{SEQ}.npy")
        if os.path.exists(path):
            continue
        d = ds[split].filter(lambda x: len(x["text"].strip()) > 0, num_proc=32)
        d = d.map(lambda b: {"ids": tok(b["text"], add_special_tokens=False)["input_ids"]},
                  batched=True, num_proc=32, remove_columns=["text"])
        flat = np.fromiter((t for ids in d["ids"] for t in ids), dtype=np.uint16)
        body = SEQ - 2
        n = len(flat) // body
        seqs = np.empty((n, SEQ), np.uint16)
        seqs[:, 0] = tok.cls_token_id
        seqs[:, 1:-1] = flat[: n * body].reshape(n, body)
        seqs[:, -1] = tok.sep_token_id
        np.save(path, seqs)
        print(split, "tokens", len(flat), "sequences", n)


def glue_and_models():
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for task in ("sst2", "mnli"):
        load_dataset("nyu-mll/glue", task)
    for name in ("bert-base-uncased", "bert-large-uncased"):
        AutoTokenizer.from_pretrained(name)
        AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    print("glue + checkpoints cached")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["cifar", "wikitext", "glue"]
    if "cifar" in steps:
        cifar()
    if "wikitext" in steps:
        wikitext()
    if "glue" in steps:
        glue_and_models()
