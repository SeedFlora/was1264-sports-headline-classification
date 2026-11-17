"""Fit one DistilBERT candidate or refit using the benchmark settings."""

import gc
import hashlib
import os
import random
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from protocol import BASE_MODEL, BASE_REVISION

SNAPSHOT_HASHES = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


class Encoded(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        example = {name: torch.tensor(values[i], dtype=torch.long) for name, values in self.tokens.items()}
        example["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return example


def verify_snapshot(path):
    for name, expected in SNAPSHOT_HASHES.items():
        try:
            with (path / name).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except FileNotFoundError as exc:
            raise ValueError(f"Missing {name} from local snapshot of {BASE_MODEL} at {BASE_REVISION}.") from exc
        if digest != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}; use {BASE_MODEL} at {BASE_REVISION}.")


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("DistilBERT training requires PyTorch with CUDA and an available NVIDIA GPU.")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False


def fit_distilbert(corpus, train, test, seed, learning_rate, snapshot, export=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, num_labels=8, local_files_only=True,
        use_safetensors=True, ignore_mismatched_sizes=True).float().cuda()
    allowed = set(tokenizer.model_input_names)

    def encode(indices):
        tokens = tokenizer(corpus.texts[indices].tolist(), truncation=True, padding=True, max_length=64)
        return Encoded({name: values for name, values in tokens.items() if name in allowed}, corpus.labels[indices])

    train_batches = DataLoader(encode(train), batch_size=8, shuffle=True,
                               generator=torch.Generator().manual_seed(seed), num_workers=0)
    test_batches = DataLoader(encode(test), batch_size=32, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = 3 * len(train_batches)
    warmup = max(1, int(0.1 * steps))

    def schedule(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    model.train()
    for _ in range(3):
        running = []
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{name: values.cuda() for name, values in batch.items()}).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError("Nonfinite gradient.")
            optimizer.step()
            scheduler.step()
            running.append(loss.item())
        losses.append(float(np.mean(running)))

    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in test_batches:
            batch.pop("labels")
            logits = model(**{name: values.cuda() for name, values in batch.items()}).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits.")
            predictions.extend(logits.argmax(-1).cpu().tolist())
    torch.cuda.synchronize()
    metrics = {"elapsed_seconds": time.perf_counter() - started, "epoch_training_loss": losses,
               "learning_rate": learning_rate, "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "device": torch.cuda.get_device_name(0)}
    if export is not None:
        model.config.id2label = dict(enumerate(corpus.class_names))
        model.config.label2id = {name: i for i, name in enumerate(corpus.class_names)}
        model.config.problem_type = "single_label_classification"
        model.save_pretrained(export / "distilbert")
        tokenizer.save_pretrained(export / "distilbert")
    del model, tokenizer, optimizer, scheduler, train_batches, test_batches, loss, grad, logits, batch
    gc.collect()
    torch.cuda.empty_cache()
    return np.asarray(predictions), metrics
