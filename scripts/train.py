"""Run the five-candidate nested benchmark on the pinned sports corpus."""

import argparse
import importlib.metadata
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import revision_lexical
from protocol import (BASE_MODEL, BASE_REVISION, CORPUS_SHA256, FOLDS, GRIDS, SEEDS,
                      checked_partitions, load_corpus, measures, read_json, split_manifest, write_json)


def fit_lexistack(corpus, train, test, seed, svm_c, export=None):
    started = time.perf_counter()
    model = revision_lexical.StrictLexicalEnsemble(seed=seed)
    with threadpool_limits(limits=1):
        model.fit(corpus.texts[train], corpus.labels[train], groups=corpus.groups[train], svm_c=svm_c)
        predictions = model.predict(corpus.texts[test])
    if export is not None:
        import joblib
        joblib.dump(model, export / "lexistack.joblib", compress=3)
    return predictions, {"elapsed_seconds": time.perf_counter() - started, "svm_C": svm_c, "base_lr_C": 4 * svm_c}


def write_predictions(path, corpus, subset, predictions):
    if len(predictions) != len(subset) or not np.isin(predictions, np.arange(8)).all():
        raise RuntimeError("Unexpected number or value of predictions.")
    pd.DataFrame({"row_id": subset, "group_id": corpus.groups[subset], "true": corpus.labels[subset],
                  "pred": predictions}).to_csv(path, index=False)
    return measures(corpus.labels[subset], predictions)


def train_fold(method, folder, corpus, split, fit, export):
    folder.mkdir(parents=True, exist_ok=True)
    validation_scores = []
    grid = GRIDS[method]
    for candidate, value in enumerate(grid):
        predictions, info = fit(corpus, split.inner_train, split.validation, split.seed, value)
        scores = write_predictions(folder / f"candidate{candidate}_predictions.csv", corpus, split.validation, predictions)
        write_json(folder / f"candidate{candidate}.json", {"candidate": candidate, **scores, **info})
        validation_scores.append(scores["f1_weighted"])
    selected = max(range(len(grid)), key=validation_scores.__getitem__)
    write_json(folder / "selection.json", {"selected_candidate": selected, "validation_scores": validation_scores})
    predictions, info = fit(corpus, split.train, split.test, split.seed, grid[selected], export=export)
    scores = write_predictions(folder / "outer_predictions.csv", corpus, split.test, predictions)
    if export is not None:
        write_json(export / "labels.json", list(corpus.class_names))
        write_json(export / "refit.json", {"method": method, "seed": split.seed, "fold": split.fold,
                                           "selected_candidate": selected, "candidate_value": grid[selected], **scores})
    record = folder / "outer_metrics.json"
    temporary_record = record.with_suffix(".tmp")
    write_json(temporary_record, {"selected_candidate": selected, **scores, **info})
    temporary_record.replace(record)
    print(f"{method} seed {split.seed} fold {split.fold} | chose {grid[selected]} | weighted F1 {scores['f1_weighted']:.6f}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Local filtered_sports_headlines.csv or sports_headlines.csv")
    parser.add_argument("--output", type=Path, required=True, help="New directory for predictions and metrics")
    parser.add_argument("--methods", nargs="+", choices=tuple(GRIDS), default=list(GRIDS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--checkpoint", type=Path, help=f"Local {BASE_MODEL} snapshot at {BASE_REVISION}")
    parser.add_argument("--export-model", type=Path, help="Save the selected refit when running one method, seed and fold")
    parser.add_argument("--resume", action="store_true", help="Retry incomplete folds with the same arguments")
    args = parser.parse_args()
    methods = [method for method in GRIDS if method in args.methods]
    seeds = [seed for seed in SEEDS if seed in args.seeds]
    folds = [fold for fold in FOLDS if fold in args.folds]
    if args.export_model is not None:
        if args.resume or len(methods) * len(seeds) * len(folds) != 1:
            parser.error("--export-model requires one method, seed and fold in a fresh run")
        if args.export_model.exists() or args.output.resolve().is_relative_to(args.export_model.resolve()):
            parser.error("--export-model must be a new directory and cannot contain --output")
    if args.output.exists() and not args.resume:
        parser.error("--output already exists; choose a new directory or use --resume")
    if args.resume and not (args.output / "run.json").is_file():
        parser.error("--resume requires a run.json in --output")

    fitters = {"LexiStack": fit_lexistack}
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
    if "DistilBERT" in methods:
        if args.checkpoint is None:
            parser.error("--checkpoint is required when training DistilBERT")
        try:
            from finetune_distilbert import configure_cuda, fit_distilbert, verify_snapshot
        except ImportError as exc:
            parser.error(f"Install requirements-distilbert.txt before training DistilBERT ({exc}).")
        try:
            verify_snapshot(args.checkpoint)
            configure_cuda()
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))

        def fit_pinned_distilbert(corpus, train, test, seed, value, export=None):
            return fit_distilbert(corpus, train, test, seed, value, args.checkpoint, export=export)

        fitters["DistilBERT"] = fit_pinned_distilbert
        packages.extend(("torch", "transformers", "huggingface-hub", "safetensors"))

    try:
        corpus = load_corpus(args.data)
        partitions = checked_partitions(corpus, seeds, folds)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    metadata = {"corpus_sha256": CORPUS_SHA256, "methods": methods, "seeds": seeds, "folds": folds,
                "class_names": list(corpus.class_names), "export_model": args.export_model is not None,
                "candidate_grid": {method: list(GRIDS[method]) for method in methods},
                "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in packages}}
    if "DistilBERT" in methods:
        metadata["base_checkpoint"] = {"repository": BASE_MODEL, "revision": BASE_REVISION}
    run_file = args.output / "run.json"
    manifest = split_manifest(corpus, partitions)
    manifest_file = args.output / "splits.csv"
    if args.resume:
        if read_json(run_file) != metadata:
            parser.error("Run metadata differs. Resume with the same methods, seeds, folds and environment.")
        if manifest_file.exists() and not pd.read_csv(manifest_file).equals(manifest):
            parser.error("Existing split manifest differs from the benchmark partitions.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(run_file, metadata)
        if args.export_model is not None:
            args.export_model.mkdir(parents=True, exist_ok=False)
    if not manifest_file.exists():
        temporary_manifest = manifest_file.with_suffix(".tmp")
        manifest.to_csv(temporary_manifest, index=False)
        temporary_manifest.replace(manifest_file)

    for method in methods:
        for split in partitions:
            folder = args.output / method / f"seed{split.seed}_fold{split.fold}"
            if args.resume and (folder / "outer_metrics.json").is_file():
                continue
            train_fold(method, folder, corpus, split, fitters[method], args.export_model)


if __name__ == "__main__":
    main()
