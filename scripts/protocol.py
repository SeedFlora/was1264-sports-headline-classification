"""Load the benchmark corpus and generate group-disjoint partitions."""

import hashlib
import html
import json
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

DATA_SHA256 = "325057b89270bda1d623931345bafaa0ac714a28dbc2d92f382c2dbbed545fda"
CORPUS_SHA256 = "581e0fa19f28fa3746db06565afef48bf0aff1ae45c4b4e572ac198aca927c0e"
GROUPS_SHA256 = "50ef54a7f5c75f40d76b1db60f49389d687d86e4a38b20c2a65de15610d5d933"
SPLITS_SHA256 = "30a4fef575f927134006e06839771af2bb410d89028adffd16bc27e110e6c2a2"
DATA_ROWS = 2833
GROUP_COUNT = 2383
CLASSES = ("Badminton", "Basketball", "Cycling", "Formula1", "Gymnastics", "Table Tennis", "Tennis", "Volleyball")
SEEDS = (42, 123, 2026)
FOLDS = (1, 2, 3, 4, 5)
GRIDS = {
    "LexiStack": (0.25, 1.0, 4.0, 0.0625, 16.0),
    "DistilBERT": (1e-5, 3e-5, 5e-5, 1e-4, 2e-4),
}
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


@dataclass(frozen=True)
class Corpus:
    texts: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class Split:
    seed: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    inner_train: np.ndarray
    validation: np.ndarray


def normalize(text):
    value = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    value = "".join(c if c.isalnum() or c.isspace() else " " for c in value)
    return " ".join(value.split())


def duplicate_groups(texts):
    normalized = [normalize(text) for text in texts]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False,
                             min_df=1, norm="l2").fit_transform(normalized)
    similarities = linear_kernel(matrix, matrix)
    left, right = np.where(np.triu(similarities >= 0.90, k=1))
    parents = list(range(len(texts)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)

    first = {}
    for i, text in enumerate(normalized):
        union(i, first.setdefault(text, i))
    for i, j in zip(left.tolist(), right.tolist()):
        union(i, j)
    return np.array([find(i) for i in range(len(texts))], dtype=np.int64)


def load_corpus(path):
    data = pd.read_csv(path)
    if data.columns.tolist() != ["Headline", "Sport"]:
        raise ValueError("Expected a CSV with Headline and Sport columns in that order.")
    if len(data) == 9985:
        data = data.loc[data.Sport.isin(CLASSES)].reset_index(drop=True)
    if len(data) != DATA_ROWS or data.isna().any().any():
        raise ValueError("Expected 2,833 complete rows after selecting the eight sports.")
    rows = json.dumps(list(data.itertuples(index=False, name=None)), ensure_ascii=False, separators=(",", ":"))
    if hashlib.sha256(rows.encode()).hexdigest() != CORPUS_SHA256:
        raise ValueError("The parsed CSV does not match the benchmark values and original row order.")
    encoder = LabelEncoder()
    labels = encoder.fit_transform(data.Sport)
    texts = data.Headline.to_numpy(dtype=object)
    groups = duplicate_groups(texts)
    group_digest = hashlib.sha256(groups.astype("<i8").tobytes()).hexdigest()
    if len(np.unique(groups)) != GROUP_COUNT or group_digest != GROUPS_SHA256:
        raise ValueError("Regenerated group IDs differ from the benchmark. Check the dependency versions.")
    return Corpus(texts, labels, groups, tuple(encoder.classes_.tolist()))


def splits(corpus, seeds=SEEDS, folds=FOLDS):
    texts, labels, groups = corpus.texts, corpus.labels, corpus.groups
    for seed in seeds:
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(outer.split(texts, labels, groups), 1):
            if fold not in folds:
                continue
            inner = StratifiedGroupKFold(5, shuffle=True, random_state=seed + 909)
            inner_train, validation = next(inner.split(texts[train], labels[train], groups[train]))
            if (set(groups[train]) & set(groups[test]) or
                    set(groups[train[inner_train]]) & set(groups[train[validation]])):
                raise RuntimeError("Generated partitions share a duplicate group.")
            yield Split(seed, fold, train, test, train[inner_train], train[validation])


def split_manifest(corpus, partitions):
    records = []
    for split in partitions:
        for role, rows in (("inner_train", split.inner_train), ("validation", split.validation),
                           ("outer_test", split.test)):
            records.extend((split.seed, split.fold, role, i, corpus.groups[i]) for i in rows)
    return pd.DataFrame(records, columns=["seed", "fold", "role", "row_id", "group_id"])


def checked_partitions(corpus, seeds=SEEDS, folds=FOLDS):
    partitions = list(splits(corpus))
    manifest = split_manifest(corpus, partitions)
    digest = hashlib.sha256(manifest.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    if digest != SPLITS_SHA256:
        raise ValueError("Generated partitions differ from the benchmark. Check the dependency versions.")
    return [split for split in partitions if split.seed in seeds and split.fold in folds]


def measures(actual, predicted):
    return {
        "f1_weighted": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
