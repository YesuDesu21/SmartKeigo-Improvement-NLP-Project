"""Single-sentence inference for the fine-tuned T5 baseline (Track B), used by the Streamlit app."""
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from training.train_t5 import fmt

DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "t5-keigo" / "best"
T5_REGISTERS = ("teineigo", "sonkeigo", "kenjogo", "no_change")


class T5Keigo:
    def __init__(self, ckpt: Path = DEFAULT_CKPT):
        self.tok = T5Tokenizer.from_pretrained(str(ckpt), is_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(str(ckpt)).eval()

    def formalize(self, text: str, register: str) -> str:
        """Same decoding settings as predict_t5.py (beam 4, max 128)."""
        ids = self.tok(fmt(register, text), return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = self.model.generate(**ids, num_beams=4, max_length=128)
        return self.tok.decode(out[0], skip_special_tokens=True)
