# DMS Contrastive Pipeline Context

## Entry points
- `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/run_pipeline.sh`
- `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/run_pipeline_parallel.sh`
- Both call `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/pipeline.py`.

## Typical run mode in this repo
- HPC Slurm jobs (`gpu4_medium`) with module `python/gpu/3.10.6-cuda12.9` and local `venv`.
- Data root: `/gpfs/scratch/jv2807/dms_data`.
- Common settings:
  - `--split_by_gene`
  - `--train_same_gene_batch` and `--test_same_gene_batch`
  - `--normalize_to_wt` with `--metadata_path`
  - `--ohe_baseline`
  - precomputed embedding H5 files (`--embeddings_path`, `--ohe_embeddings_path`)

## Shell scripts behavior

### `run_pipeline.sh`
- Runs sequentially over `Stability`, `OrganismalFitness`.
- Uses one GPU.
- Uses `--num_epochs 0` + explicit `--model_path`, so this is evaluation-only mode.
- Uses split file from an earlier run path under `results/.../data_split.json`.

### `run_pipeline_parallel.sh`
- Runs 3 background jobs in one Slurm allocation, each pinned with `CUDA_VISIBLE_DEVICES=$i`.
- Selection types: `Activity`, `OrganismalFitness`, `Binding`.
- Uses `--use_lora` and trains (`--num_epochs 3`).

## `pipeline.py` architecture

### High-level flow (`main`)
1. Parse CLI args and set global config.
2. Load model/tokenizer:
   - ESM2 (`AutoModel`) if model name contains `esm2`
   - ESM-C (`AutoModelForMaskedLM`, hardcoded `Synthyra/ESMplusplus_large`) if contains `esmc`
3. Optional LoRA wrapping (`scripts/lora_utils.py`) when `--use_lora`.
4. Load and preprocess dataset:
   - Drops missing rows
   - Per-gene quartile binning into `high`/`low`
5. Build train/test split (gene-based by default; can use split JSON).
6. Build custom `DMSContrastiveDataset` + custom `DataLoader` (balanced high/low batching; optional gene-aware batching).
7. Build projection head (`ContrastiveNetwork`) + contrastive objective (`ContrastiveLoss`).
8. Train or load checkpoint.
9. Optional OHE+LLR baseline workflow:
   - precompute batches
   - compare baseline vs learned projections across train-size fractions
10. Close H5 embedding loaders.

## Embedding and cache subsystem
- `scripts/h5_utils.py` `EmbeddingLoader` stores embeddings in H5 (`X`, `seq_ids`) + sidecar JSON hash index.
- `pipeline.py` `load_embeddings_h5(...)`:
  - loads by sequence hash
  - backfills missing embeddings via ESM forward pass (`esm_batch`) or OHE/LLR feature generation.

## OHE + LLR baseline path
- Enabled by `--ohe_baseline`.
- Uses `esm-variants` repo (path via `--esm_variants_module_path`) to compute LLR scores.
- `get_ohe_features(...)` creates:
  - first column = LLR score
  - remaining columns = flattened one-hot sequence (WT mutated at variant positions)
- Metrics from `scripts/metrics.py`:
  - KNN, Ridge classification
  - optional ridge regression mode via `--eval_regression`

## Important files/functions
- Pipeline core: `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/pipeline.py`
- H5 loader: `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/scripts/h5_utils.py`
- LoRA helpers: `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/scripts/lora_utils.py`
- Metrics: `/Users/jv2807/Desktop/brandes_lab/dms_contrastive/scripts/metrics.py`

## Outputs
- Run directory: `results/<run_name>/`
- Common artifacts:
  - `args.json`
  - `data_split.json`
  - `model.pt` / `temp_model.pt`
  - plots and result dict JSON files
- Aggregate CSV: `results/results.csv`

## Fast mental model for future sessions
- This project is a contrastive projection-head training/eval pipeline over mutation embeddings.
- Embeddings are usually precomputed and cached in H5; LoRA mode can produce embeddings online from ESM.
- Main evaluation emphasis in current scripts is baseline comparison (`--ohe_baseline`) and split-by-gene generalization.
- Most runs are configured through Slurm shell wrappers, not ad-hoc CLI invocations.
