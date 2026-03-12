#!/bin/bash
#SBATCH -p gpu8_medium
#SBATCH --gres=gpu:3
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j_lora.out
#SBATCH --job-name=dms_cl

SINGULARITY="singularity exec --nv --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

BASE_DATA_PATH="/scratch/jv2807/dms_data"
EMBEDDING_LAYER="layer33_mean"

SELECTION_TYPES=("Stability" "OrganismalFitness" "Binding")

# Manually assign each task to a specific GPU
for i in "${!SELECTION_TYPES[@]}"; do
    COARSE_SELECTION_TYPE="${SELECTION_TYPES[$i]}"
    echo "Launching ${COARSE_SELECTION_TYPE} on GPU ${i}"
    
    RUN_NAME="650M_esm2_lora_NWT_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"
    
    CUDA_VISIBLE_DEVICES=$i $SINGULARITY /bin/bash -c "source /ext3/env.sh; cd /home/jv2807/dms_contrastive && CUDA_VISIBLE_DEVICES=$i python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33.h5 \
        --ohe_embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/ohe_embeddings.h5 \
        --model_cache /scratch/jv2807/cache \
        --model_name esm2 \
        --esm_max_length 600 \
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
        --lora_target_modules query key value \
        --split_by_gene \
        --split_file /home/jv2807/dms_contrastive/results/650M_splitbygene_lora2_${COARSE_SELECTION_TYPE}_layer33_mean/data_split.json \
        --ohe_baseline \
        " > logs/${SLURM_JOB_ID}_${COARSE_SELECTION_TYPE}.out 2>&1 &
done

wait

#for esm-c  --lora_target_modules layernorm_qkv.1 out_proj \
#for esm2 key, value, target



echo "All tasks completed"