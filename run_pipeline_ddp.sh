#!/bin/bash
#SBATCH --account=torch_pr_800_cds
#SBATCH --gres=gpu:a100:4          # <-- change GPU count here (must match NGPUS below)
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH --output=logs/%j.out
#SBATCH --job-name=dms_ddp
#SBATCH --signal=B:TERM@60

NGPUS=4   # must match --gres above

SINGULARITY="singularity exec --nv --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

BASE_DATA_PATH="/scratch/jv2807/dms_data"

RUN_NAME="all_selection_types_ddp"

$SINGULARITY /bin/bash -c "source /ext3/env.sh; cd /home/jv2807/dms_contrastive && torchrun --standalone --nproc_per_node=$NGPUS pipeline.py \
    --run_name $RUN_NAME \
    --data_path $BASE_DATA_PATH/datasets/all_selection_types.csv \
    --embeddings_path $BASE_DATA_PATH/embeddings/{selection_type}/600M_esmc_mean.h5 \
    --ohe_embeddings_path $BASE_DATA_PATH/embeddings/{selection_type}/ohe_embeddings.h5 \
    --model_cache /scratch/jv2807/cache \
    --model_name esmc \
    --esm_max_length 600 \
    --batch_size 256 \
    --gradient_accumulation_steps 1 \
    --patience 6 \
    --eval_per_epoch 2 \
    --dropout 0.0 \
    --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
    --num_epochs 1 \
    --train_same_gene_batch \
    --test_same_gene_batch \
    --normalize_to_wt \
    --split_by_gene \
    --ohe_baseline \
    --num_bootstraps 4 \
    --selection_types Stability OrganismalFitness Binding Activity Expression \
    --lora_alpha 32 \
    --esm_lr .000005 \
    --lora_target_modules layernorm_qkv.1"
