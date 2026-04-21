"""
dataset.py — Dataset loading and tokenization for training.

Converts raw JSONL distillation data into tokenized training examples
suitable for the SFT (Supervised Fine-Tuning) training loop.

Training format:
  The model learns to predict the teacher's response given the question.
  Loss is only computed on the response tokens (not the question tokens)
  using the standard -100 label masking trick.

  [SYSTEM PROMPT] [USER: question] [ASSISTANT: <think>...</think><answer>...</answer>]
   ↑                ↑                ↑
   ignored          ignored          ← loss computed here

This is standard SFT / instruction fine-tuning practice.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from pathlib import Path
from typing import Optional
from rich.console import Console

from .config import CFG
from .tool_loop import build_system_prompt

console = Console()


class CoTDataset(Dataset):
    """PyTorch Dataset for Chain-of-Thought distillation data.

    Reads a JSONL file where each line is:
    {
        "question": "...",
        "teacher_response": "<think>...</think><answer>...</answer>",
        ...
    }

    And tokenizes it into input_ids + labels for causal LM training.

    Args:
        jsonl_path: Path to the JSONL dataset file.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: Maximum token length (truncate longer sequences).
        split: "train" or "eval" (eval uses last 10% of data).
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = None,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len or CFG.model.max_seq_len
        self.system_prompt = build_system_prompt()

        # Load all records
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Split: 90% train, 10% eval
        n = len(records)
        if split == "train":
            self.records = records[:int(n * 0.9)]
        else:
            self.records = records[int(n * 0.9):]

        console.print(
            f"[green]✓ Loaded {len(self.records)} {split} examples[/] "
            f"from {jsonl_path}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Tokenize a single example.

        Returns:
            dict with:
              "input_ids": (seq_len,) token ids
              "attention_mask": (seq_len,) 1/0 mask
              "labels": (seq_len,) target ids (-100 where loss is ignored)
        """
        record = self.records[idx]
        question = record["question"]
        response = record["teacher_response"]

        # Build the full conversation in chat format
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Tokenize the prompt (input part — no labels here)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Adds <|im_start|>assistant\n
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]

        # Tokenize the response (target part — compute loss here)
        response_ids = self.tokenizer(
            response + self.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]

        # Concatenate: [prompt] + [response]
        input_ids = torch.cat([prompt_ids, response_ids])

        # Labels: -100 for prompt tokens (ignored in loss), actual ids for response
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            response_ids,
        ])

        # Truncate if too long
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader: pad sequences to same length.

    PyTorch requires all tensors in a batch to have the same shape.
    We pad shorter sequences to the length of the longest in the batch.

    Padding:
    - input_ids: padded with tokenizer.pad_token_id
    - attention_mask: padded with 0 (ignore padding)
    - labels: padded with -100 (ignore in loss)
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Right-pad sequences
        padded_input_ids.append(
            torch.cat([
                item["input_ids"],
                torch.zeros(pad_len, dtype=torch.long),  # pad_token_id=0
            ])
        )
        padded_attention_masks.append(
            torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
        )
        padded_labels.append(
            torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ])
        )

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.stack(padded_labels),
    }


def build_dataloaders(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizer,
) -> tuple[DataLoader, DataLoader]:
    """Build train and eval DataLoaders.

    Args:
        jsonl_path: Path to JSONL dataset.
        tokenizer: Tokenizer.

    Returns:
        (train_loader, eval_loader) tuple.
    """
    train_dataset = CoTDataset(jsonl_path, tokenizer, split="train")
    eval_dataset = CoTDataset(jsonl_path, tokenizer, split="eval")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,      # 0 workers on Windows to avoid multiprocessing issues
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=CFG.training.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    console.print(
        f"[green]✓ DataLoaders ready[/] — "
        f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}"
    )
    return train_loader, eval_loader
