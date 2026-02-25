#!/bin/bash
#SBATCH -p gpu4_medium
#SBATCH --gres=gpu:3
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j_lora.out
#SBATCH --job-name=dms_cl

module load python/gpu/3.10.6-cuda12.9
source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jv2807/dms_data"
EMBEDDING_LAYER="layer33_mean"

SELECTION_TYPES=("Activity" "OrganismalFitness" "Binding")

# Manually assign each task to a specific GPU
for i in "${!SELECTION_TYPES[@]}"; do
    COARSE_SELECTION_TYPE="${SELECTION_TYPES[$i]}"
    echo "Launching ${COARSE_SELECTION_TYPE} on GPU ${i}"
    
    RUN_NAME="600M_esmc_lora_NWT_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"
    
    CUDA_VISIBLE_DEVICES=$i python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/600M_esmc_mean.h5 \
        --ohe_embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/ohe_embeddings.h5 \
        --model_cache /gpfs/scratch/jv2807/cache \
        --model_name esmc \
        --esm_max_length 600 \
        --input_dim 1152 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --patience 4 \
        --eval_per_epoch 2 \
        --dropout 0.0 \
        --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
        --num_epochs 3 \
        --train_same_gene_batch \
        --test_same_gene_batch \
        --normalize_to_wt \
        --use_lora \
        --lora_alpha 32 \
        --esm_lr .000005 \
        --lora_target_modules layernorm_qkv.1 out_proj \
        --split_by_gene \
        --split_file /gpfs/home/jv2807/dms_contrastive/results/650M_splitbygene_lora2_${COARSE_SELECTION_TYPE}_layer33_mean/data_split.json \
        --ohe_baseline \
        > logs/${SLURM_JOB_ID}_${COARSE_SELECTION_TYPE}.out 2>&1 &
done

wait

#for esm-c  --lora_target_modules layernorm_qkv.1 out_proj \
#for esm2 key, value, target



echo "All tasks completed"