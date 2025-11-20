"""Shared helpers for the FSDP training demos."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def setup_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@contextmanager
def _main_process_first(is_main_process: bool):
    """Ensure rank-0 runs the protected block before other ranks."""

    distributed_ready = dist.is_available() and dist.is_initialized()
    if not distributed_ready:
        yield
        return

    if not is_main_process:
        dist.barrier()

    try:
        yield
    finally:
        dist.barrier()


def load_tinystories(tokenizer: AutoTokenizer, seq_len: int, *, is_main_process: bool) -> Dataset:
    """Tokenize and pack TinyStories into contiguous blocks for LM training."""

    with _main_process_first(is_main_process):
        raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=seq_len + 1,
            return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    with _main_process_first(is_main_process):
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def pack_sequences(examples):
        flat_tokens = []
        for input_ids in examples["input_ids"]:
            flat_tokens.extend(input_ids)

        num_sequences = len(flat_tokens) // (seq_len + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (seq_len + 1)
            end_idx = start_idx + (seq_len + 1)
            sequence = flat_tokens[start_idx:end_idx]
            packed_input_ids.append(sequence[:-1])
            packed_labels.append(sequence[1:])

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    with _main_process_first(is_main_process):
        packed_dataset = tokenized_dataset.map(
            pack_sequences,
            batched=True,
            remove_columns=tokenized_dataset.column_names,
            batch_size=1000,
        )

    return packed_dataset.shuffle(seed=2025)


def create_collate_fn():
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    return collate_fn


def get_model_flops_per_token(config, seq_len: int) -> float:
    """Estimate FLOPs/token using a rough decoder-only formula."""
    head_dim = config.hidden_size // config.num_attention_heads
    mlp = 18 * config.hidden_size * config.intermediate_size
    attn_proj = 12 * head_dim * (config.num_attention_heads + config.num_key_value_heads)
    attn_scores = 12 * config.num_attention_heads * head_dim * seq_len
    return (mlp + attn_proj + attn_scores) * config.num_hidden_layers


class ThroughputTracker:
    """Tracks warmup and steady-state throughput plus peak memory."""

    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        self.start_time = None
        self.tokens = 0
        self.steps_seen = 0
        self.in_warmup = True

    def step(self, tokens: int, flops_per_token: float | None = None) -> dict:
        self.steps_seen += 1

        if self.steps_seen == self.warmup_steps:
            self.start_time = time.perf_counter()
            self.tokens = 0
            self.in_warmup = False
            return {"warmup_done": True}

        if self.in_warmup or self.start_time is None:
            return {}

        self.tokens += tokens
        elapsed = time.perf_counter() - self.start_time
        steps = self.steps_seen - self.warmup_steps
        metrics = {
            "tokens_per_second": self.tokens / elapsed,
            "steps_per_second": steps / elapsed,
            "total_tokens": self.tokens,
            "total_time": elapsed,
        }

        if flops_per_token is not None:
            metrics["tflops_per_device"] = (flops_per_token * self.tokens) / (elapsed * 1e12)

        return metrics

    @staticmethod
    def format(metrics: dict, include_memory: bool = False) -> str:
        msg = (
            f" | steps/s={metrics['steps_per_second']:.2f}"
            f" | toks/s={metrics['tokens_per_second']:.0f}"
        )
        if "tflops_per_device" in metrics:
            msg += f" | TFLOPs/rank={metrics['tflops_per_device']:.2f}"
        if include_memory:
            msg += (
                f" | peak_active={metrics.get('peak_memory_active', 0):.2f}GB"
                f" | peak_reserved={metrics.get('peak_memory_reserved', 0):.2f}GB"
            )
        return msg


def gpu_memory_usage(device=0):
    """Return a small dict of peak memory stats and reset counters."""
    backend = torch.cuda
    dev = torch.device(f"cuda:{device}")
    gib = 1024**3
    stats = backend.memory_stats(dev)
    mem = {
        "peak_memory_active": stats.get("active_bytes.all.peak", 0) / gib,
        "peak_memory_alloc": backend.max_memory_allocated(dev) / gib,
        "peak_memory_reserved": backend.max_memory_reserved(dev) / gib,
    }
    backend.reset_peak_memory_stats(dev)
    return mem
