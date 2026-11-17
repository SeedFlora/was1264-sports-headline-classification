"""Verify stored predictions, check model selection and recompute held-out metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from protocol import (CORPUS_SHA256, FOLDS, GRIDS, SEEDS, checked_partitions, load_corpus,
                      measures, read_json, split_manifest, write_json)


def verify_predictions(path, corpus, subset, record):
    frame = pd.read_csv(path)
    if (not np.array_equal(frame.row_id, subset) or
            not np.array_equal(frame.group_id, corpus.groups[subset]) or
            not np.array_equal(frame.true, corpus.labels[subset]) or
            not np.isin(frame.pred, np.arange(8)).all()):
        raise ValueError(f"Prediction rows, labels or values do not match the assigned partition: {path}")
    computed = measures(frame.true, frame.pred)
    if any(not np.isclose(record[key], value, rtol=0, atol=1e-12) for key, value in computed.items()):
        raise ValueError(f"Stored metric differs from its prediction file: {path}")
    return frame, computed


def verify_fold(folder, corpus, subset, method):
    grid = GRIDS[method]
    candidate_scores = []
    key = "svm_C" if method == "LexiStack" else "learning_rate"
    for candidate, value in enumerate(grid):
        record = read_json(folder / f"candidate{candidate}.json")
        if record["candidate"] != candidate or record[key] != value:
            raise ValueError(f"Unexpected candidate configuration: {folder}")
        _, computed = verify_predictions(folder / f"candidate{candidate}_predictions.csv", corpus, subset.validation, record)
        candidate_scores.append(computed["f1_weighted"])

    selected = max(range(len(grid)), key=candidate_scores.__getitem__)
    selection = read_json(folder / "selection.json")
    chosen_scores = selection["validation_scores"]
    if (selection["selected_candidate"] != selected or len(chosen_scores) != len(candidate_scores) or
            not np.allclose(chosen_scores, candidate_scores, rtol=0, atol=1e-12)):
        raise ValueError(f"Stored winner differs from validation selection: {folder}")
    record = read_json(folder / "outer_metrics.json")
    if record["selected_candidate"] != selected or record[key] != grid[selected]:
        raise ValueError(f"Outer refit used a different candidate: {folder}")
    frame, metrics = verify_predictions(folder / "outer_predictions.csv", corpus, subset.test, record)
    return frame, {"seed": subset.seed, "fold": subset.fold, "candidate_value": grid[selected], **metrics}


def weighted_f1(confusion):
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    by_class = np.divide(2 * true_positive, support + predicted, out=np.zeros_like(support, dtype=float),
                         where=(support + predicted) != 0)
    return np.sum(by_class * support, axis=-1) / support.sum(axis=-1)


def paired_interval(corpus, predictions, seeds):
    _, position = np.unique(corpus.groups, return_inverse=True)
    n_groups = int(position.max() + 1)
    tables = []
    for method in ("LexiStack", "DistilBERT"):
        tables_per_seed = []
        for seed in seeds:
            frame = predictions[method, seed]
            confusion = np.zeros((n_groups, 8, 8), dtype=np.int16)
            np.add.at(confusion, (position, frame.true, frame.pred), 1)
            tables_per_seed.append(confusion)
        tables.append(np.stack(tables_per_seed))

    rng = np.random.default_rng(20261030)
    probabilities = np.repeat(1 / n_groups, n_groups)
    differences = []
    with threadpool_limits(limits=1):
        for _ in range(50):
            weights = rng.multinomial(n_groups, probabilities, size=100)
            aggregates = [np.einsum("bg,sgij->bsij", weights, table, optimize=True) for table in tables]
            lexical, distilbert = [100 * weighted_f1(matrix).mean(axis=1) for matrix in aggregates]
            differences.extend(lexical - distilbert)
    return {"interval95_pp": np.quantile(differences, [.025, .975]).tolist(), "bootstrap_draws": 5000,
            "resampled_groups": n_groups, "rng_seed": 20261030,
            "scope": "Conditional on fitted models, evaluated seeds, and the original corpus."}


def evaluate_run(run, corpus, methods, seeds, folds, partitions):
    complete = tuple(folds) == FOLDS
    report = {"methods": {}, "seeds": list(seeds), "folds": list(folds), "verified_top_level_fits": 0}
    saved_predictions = {}
    for method in methods:
        summary = {"folds": []}
        by_seed = {seed: [] for seed in seeds}
        for split in partitions:
            folder = run / method / f"seed{split.seed}_fold{split.fold}"
            frame, metrics = verify_fold(folder, corpus, split, method)
            report["verified_top_level_fits"] += len(GRIDS[method]) + 1
            summary["folds"].append(metrics)
            by_seed[split.seed].append(frame)
        if complete:
            pooled = []
            for seed in seeds:
                frame = pd.concat(by_seed[seed]).sort_values("row_id").reset_index(drop=True)
                if not np.array_equal(frame.row_id, np.arange(len(corpus.labels))):
                    raise ValueError(f"Missing rows or duplicate rows for {method} seed {seed}")
                saved_predictions[method, seed] = frame
                pooled.append({"seed": seed, **measures(frame.true, frame.pred)})
            summary["pooled_out_of_fold_per_seed"] = pooled
            summary["mean_pooled_weighted_f1"] = float(np.mean([row["f1_weighted"] for row in pooled]))
        report["methods"][method] = summary
    if complete and set(methods) == set(GRIDS):
        comparison = paired_interval(corpus, saved_predictions, seeds)
        scores = report["methods"]
        comparison["weighted_f1_gap_pp"] = 100 * (scores["LexiStack"]["mean_pooled_weighted_f1"] -
                                                 scores["DistilBERT"]["mean_pooled_weighted_f1"])
        report["lexistack_minus_distilbert"] = comparison
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = read_json(args.run / "run.json")
        corpus = load_corpus(args.data)
        methods, seeds, folds = (metadata[name] for name in ("methods", "seeds", "folds"))
        if (metadata["corpus_sha256"] != CORPUS_SHA256 or metadata["class_names"] != list(corpus.class_names)
                or not methods or methods != [item for item in GRIDS if item in methods]
                or not seeds or seeds != [item for item in SEEDS if item in seeds]
                or not folds or folds != [item for item in FOLDS if item in folds]
                or metadata["candidate_grid"] != {method: list(GRIDS[method]) for method in methods}):
            raise ValueError("Run metadata does not match the benchmark configuration.")
        partitions = checked_partitions(corpus, seeds, folds)
        if not pd.read_csv(args.run / "splits.csv").equals(split_manifest(corpus, partitions)):
            raise ValueError("Run split manifest does not match the benchmark partitions.")
        report = evaluate_run(args.run, corpus, methods, seeds, folds, partitions)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    write_json(args.run / "evaluation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
