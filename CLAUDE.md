# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment (NYU Torch HPC)

Singularity + conda overlay (no module load needed):

```bash
# Overlay: /scratch/jv2807/dms_singularity/dms_contrastive.ext3
# Image:   /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# Interactive use:
singularity exec --nv \
  --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; python ..."
```

Data root: `/gpfs/scratch/jv2807/dms_data`
Results: `/gpfs/home/jv2807/dms_contrastive/results/`
Model cache: `/gpfs/scratch/jv2807/cache`

## Running the pipeline

All runs go through `pipeline.py`. Jobs are submitted via Slurm:

```bash
sbatch run_pipeline.sh          # sequential eval, 1 GPU
sbatch run_pipeline_parallel.sh # 3 parallel training jobs, 1 GPU each
sbatch run_pipeline_single.sh   # single run, eval mode
```

Key flags:
- `--num_epochs 0` = eval-only (load checkpoint via `--model_path`)
- `--use_lora` = LoRA fine-tuning of ESM backbone
- `--ohe_baseline` = run OHE+LLR baseline comparison
- `--split_by_gene` = gene-held-out train/test split (default mode)
- `--split_file <path>` = reuse a previous run's split

Selection types: `Stability`, `OrganismalFitness`, `Binding`, `Activity`, `Expression`

## Architecture

### `pipeline.py` (core)
1. Loads ESM2 (`AutoModel`) or ESM-C (`Synthyra/ESMplusplus_large`) backbone
2. Optionally wraps with LoRA (`scripts/lora_utils.py`)
3. Loads CSV dataset → per-gene quartile binning into `high`/`low` labels
4. Builds gene-aware train/test split
5. `DMSContrastiveDataset` + custom `DataLoader` with balanced high/low batching
6. `ContrastiveNetwork` (projection head) + `ContrastiveLoss`
7. Train or load checkpoint, then evaluate
8. Optional OHE+LLR baseline: compares across train-size fractions

### Key scripts
- `scripts/h5_utils.py` — `EmbeddingLoader`: stores/retrieves embeddings from H5 files by sequence hash. Missing embeddings are backfilled via ESM forward pass or OHE/LLR generation.
- `scripts/lora_utils.py` — LoRA wrapping/loading helpers
- `scripts/metrics.py` — KNN, Ridge classification/regression, contrastive metrics
- `scripts/tune_asha.py` — hyperparameter tuning

### Embeddings
- Precomputed and cached as H5 files at `$BASE_DATA_PATH/embeddings/{selection_type}/*.h5`
- ESM-C embeddings: `600M_esmc_mean.h5`, OHE: `ohe_embeddings.h5`
- ESM2 embeddings: `650M_t33_mean_layer33.h5`
- `--embeddings_path` supports `{selection_type}` as a template placeholder

### Outputs (per run in `results/<run_name>/`)
- `args.json`, `data_split.json`
- `model.pt` / `temp_model_epoch{N}.pt`
- Plots and result JSON files
- Aggregate: `results/results.csv`

## Models
- ESM2: `facebook/esm2_t33_650M_UR50D` — `--input_dim 1280`
- ESM-C: `esmc` (resolves to `Synthyra/ESMplusplus_large`) — `--input_dim 1152`
- LoRA targets: ESM2 → `query key value`; ESM-C → `layernorm_qkv.1 out_proj`
