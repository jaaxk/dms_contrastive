#!/bin/bash
#SBATCH -p gpu8_medium
#SBATCH --gres=gpu:5
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_lora.out
#SBATCH --job-name=dms_cl

module load python/gpu/3.10.6-cuda12.9
source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jv2807/dms_data"
EMBEDDING_LAYER="layer33_mean"

SELECTION_TYPES=("Stability" "OrganismalFitness" "Binding" "Activity" "Expression")

# Manually assign each task to a specific GPU
for i in "${!SELECTION_TYPES[@]}"; do
    COARSE_SELECTION_TYPE="${SELECTION_TYPES[$i]}"
    echo "Launching ${COARSE_SELECTION_TYPE} on GPU ${i}"
    
    RUN_NAME="650M_splitbygene_lora2_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"
    
    CUDA_VISIBLE_DEVICES=$i python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33.h5 \
        --ohe_embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/ohe_embeddings.h5 \
        --model_cache /gpfs/scratch/jv2807/cache \
        --model_name facebook/esm2_t33_650M_UR50D \
        --esm_max_length 600 \
        --input_dim 1280 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --patience 10 \
        --eval_batches_during_training 2000 \
        --dropout 0.0 \
        --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
        --num_epochs 10 \
        --normalize_to_wt \
        --ohe_baseline \
        --split_by_gene \
        --use_lora \
        > logs/${SLURM_JOB_ID}_${COARSE_SELECTION_TYPE}.out 2>&1 &
done

wait

echo "All tasks completed"