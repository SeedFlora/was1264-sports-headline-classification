# WAS 2026 #1264 · Sports headline classification

This repository contains the preprocessing, model selection, training, and evaluation code for LexiStack and DistilBERT in the five-candidate experiment. The task is to classify a predominantly English sports headline into one of eight sports.

[Try both classifiers in your browser](https://huggingface.co/spaces/budijuarto/was1264-sports-headline-demo) or [download the refit checkpoints](https://huggingface.co/budijuarto/was1264-sports-headline-models).

The archived experiment reported mean pooled out-of-fold weighted F1 of **96.14% for LexiStack** and **95.11% for DistilBERT**. Each model uses three seeds (42, 123, 2026) and five group-disjoint outer folds per seed. In each fold, it scores five candidates on the same inner validation split and refits the winner on the full outer training fold. Ties go to the first candidate in the published order.

| Method | Candidate grid, in selection order |
| --- | --- |
| LexiStack | Shared C scale of `0.25, 1, 4, 0.0625, 16`. Base logistic regression uses four times the scale |
| DistilBERT | Learning rate of `1e-5, 3e-5, 5e-5, 1e-4, 2e-4`. Each fit uses three epochs, FP32, 10% warm-up, and linear learning-rate decay |

## Get the data

Obtain `filtered_sports_headlines.csv` through the data location documented in the upstream repository and use the original row order. You can also pass the original 9,985-row `sports_headlines.csv` to either script. It selects the eight target sports and verifies the resulting rows against the benchmark. Keep the CSV outside version control. The benchmark uses the `Headline` and `Sport` columns, 2,833 rows, and eight classes (Badminton, Basketball, Cycling, Formula1, Gymnastics, Table Tennis, Tennis, Volleyball).

Expected file SHA-256 for the archival run:

```text
325057b89270bda1d623931345bafaa0ac714a28dbc2d92f382c2dbbed545fda
```

[Upstream data pointer at commit `5aaaec6`](https://github.com/ecehceh/NewsClassification/blob/5aaaec649af83576b8e3448c24e58b496dbcd415/Dataset/Dataset_Location.txt)

## Install the environment

The original run used Python 3.13.5, PyTorch 2.11.0 with CUDA 12.8, and an NVIDIA RTX 3070 Ti Laptop GPU. LexiStack trains on CPU. The archival DistilBERT recipe requires a CUDA-capable GPU and a local copy of the pinned pretrained model.

```bash
git clone https://github.com/SeedFlora/was1264-sports-headline-classification.git
cd was1264-sports-headline-classification
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell. Then install the dependencies:

```bash
python -m pip install -r requirements-distilbert.txt
```

For a CPU-only LexiStack run, install `requirements.txt` instead. To train DistilBERT, download the base model once. The training script uses this snapshot from disk.

```bash
hf download distilbert/distilbert-base-uncased config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt --revision 12040accade4e8a0f71eabdb258fecc2e7e948be --local-dir checkpoints/distilbert-base-uncased
```

## Train and evaluate

Pass the path to your copy of the corpus.

```bash
python scripts/train.py --data /path/to/filtered_sports_headlines.csv --output runs/five-candidate --checkpoint checkpoints/distilbert-base-uncased
python scripts/evaluate.py --data /path/to/filtered_sports_headlines.csv --run runs/five-candidate
```

For a quick check, run one LexiStack fold. Use a separate output directory for this run.

```bash
python scripts/train.py --data /path/to/filtered_sports_headlines.csv --output runs/lexical-smoke --methods LexiStack --seeds 42 --folds 1
```

The complete run records 180 top-level fits across two methods. Evaluation pools the outer predictions across five folds within each seed, averages the three seed scores, and writes `evaluation.json` to the run directory. The paired interval resamples headline groups. GPU and software differences can change the predictions. The full paper also compares five other encoders.

Add `--export-model checkpoints/lexistack-fold1` to the one-fold command above to save the selected LexiStack refit. For DistilBERT, use `--methods DistilBERT`, supply `--checkpoint`, and choose new output and export directories. The public model files are the selected refits for seed 42, fold 1. [Download checkpoints and provenance](https://huggingface.co/budijuarto/was1264-sports-headline-models).

## Source and attribution

Implementation and experiment associated with Joevan Philbert Morrow Yohansha, Christopher Evan Wijaksana, James Ko, Eduardo Richie Imanuell, and Budi Juarto (WE ARE Student 2026, paper #1264). See the upstream repository for the corpus source and [DistilBERT on Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased) for base model attribution.
