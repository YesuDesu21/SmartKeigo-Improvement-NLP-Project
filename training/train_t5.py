"""Supervised fine-tuning of sonoisa/t5-base-japanese on casual -> keigo pairs (B2 baseline).
    pip install torch transformers sentencepiece accelerate
    python train_t5.py --splits ../corpus-dataset/splits --out checkpoints/t5-keigo
    With human-reviewed generated pairs added to the training set (val/test are unchanged):
    python train_t5.py --extra generated/generated_train.jsonl --out checkpoints/t5-keigo-aug
Input format: "敬語[<register>]: <casual sentence>"  (register tag = controlled track)."""
import argparse, json, random
import torch
from torch.utils.data import Dataset
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5ForConditionalGeneration, T5Tokenizer)

BASE = "sonoisa/t5-base-japanese"


def fmt(reg, src):
    return f"敬語[{reg}]: {src}"


def load(path, no_change_frac):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["sft_core"]]
    keep, rng = [], random.Random(42)
    for r in rows:
        if r["register"] == "no_change" and rng.random() > no_change_frac:
            continue
        keep.append(r)
    return keep


class DS(Dataset):
    def __init__(self, rows, tok, max_len=128):
        self.enc = [(tok(fmt(r["register"], r["source_text"]), max_length=max_len, truncation=True)["input_ids"],
                     tok(r["target_text"], max_length=max_len, truncation=True)["input_ids"]) for r in rows]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        x, y = self.enc[i]
        return {"input_ids": x, "attention_mask": [1] * len(x), "labels": y}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="../corpus-dataset/splits")
    ap.add_argument("--out", default="checkpoints/t5-keigo")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[], help="extra training jsonl files (e.g. reviewed generated pairs)")
    ap.add_argument("--no-change-frac", type=float, default=0.3, help="share of no_change controls kept in training")
    a = ap.parse_args()
    torch.manual_seed(42)
    tok = T5Tokenizer.from_pretrained(BASE, is_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(BASE)
    train_rows = load(f"{a.splits}/train.jsonl", a.no_change_frac)
    n_corpus = len(train_rows)
    for p in a.extra:
        train_rows += load(p, 1.0)
    print(f"training rows: {n_corpus} from the corpus + {len(train_rows) - n_corpus} extra")
    tr, va = DS(train_rows, tok), DS(load(f"{a.splits}/val.jsonl", 1.0), tok)
    args = Seq2SeqTrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        weight_decay=0.01, warmup_ratio=0.1, seed=42, report_to="none", bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    Seq2SeqTrainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                   data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()
    model.save_pretrained(a.out + "/best"); tok.save_pretrained(a.out + "/best")
    print("saved", a.out + "/best")
