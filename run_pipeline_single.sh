#!/bin/bash
#SBATCH -p gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/%j_all.out
#SBATCH --job-name=dms_all
#SBATCH --signal=B:TERM@60

module load python/gpu/3.10.6-cuda12.9
source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jv2807/dms_data"

EMBEDDING_LAYER="layer33_mean"

RUN_NAME="all_selection_types_600M_esmc_NWT"

python -u pipeline.py --run_name $RUN_NAME \
    --data_path $BASE_DATA_PATH/datasets/all_selection_types.csv \
    --embeddings_path $BASE_DATA_PATH/embeddings/{selection_type}/600M_esmc_mean.h5 \
    --ohe_embeddings_path $BASE_DATA_PATH/embeddings/{selection_type}/ohe_embeddings.h5 \
    --model_cache /gpfs/scratch/jv2807/cache \
    --model_name esmc \
    --esm_max_length 600 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --patience 8 \
    --eval_per_epoch 8 \
    --dropout 0.0 \
    --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
    --num_epochs 2 \
    --train_same_gene_batch \
    --test_same_gene_batch \
    --normalize_to_wt \
    --split_by_gene \
    --ohe_baseline \
    --selection_types Stability OrganismalFitness Binding Activity Expression


#        --use_lora \
#--model_name facebook/esm2_t33_650M_UR50D \
#--embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33.h5 \
#        --use_lora \

#"Stability" "OrganismalFitness" "Binding" "Activity" "Expression" 