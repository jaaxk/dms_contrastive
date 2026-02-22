#!/bin/bash
#SBATCH -p gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/%j_lora.out
#SBATCH --job-name=dms_cl_sp

module load python/gpu/3.10.6-cuda12.9
source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jv2807/dms_data"

EMBEDDING_LAYER="layer33_mean"

for COARSE_SELECTION_TYPE in "Stability" "Expression" "Activity" "Binding" "OrganismalFitness" ; do
    echo "Running ${COARSE_SELECTION_TYPE} ${EMBEDDING_LAYER}"

    RUN_NAME="650M_NWT_nolora_evalregression_${COARSE_SELECTION_TYPE}"

    python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33_2.h5 \
        --ohe_embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/ohe_embeddings18508745    .h5 \
        --model_cache /gpfs/scratch/jv2807/cache \
        --model_name facebook/esm2_t33_650M_UR50D \
        --esm_max_length 600 \
        --input_dim 1280 \
        --batch_size 64 \
        --gradient_accumulation_steps 8 \
        --patience 4 \
        --eval_per_epoch 2 \
        --dropout 0.0 \
        --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
        --num_epochs 0 \
        --train_same_gene_batch \
        --test_same_gene_batch \
        --normalize_to_wt \
        --lora_alpha 32 \
        --esm_lr .000005 \
        --lora_target_modules query key value \
        --split_by_gene \
        --split_file /gpfs/home/jv2807/dms_contrastive/results/650M_splitbygene_lora2_${COARSE_SELECTION_TYPE}_layer33_mean/data_split.json \
        --model_path /gpfs/home/jv2807/dms_contrastive/results/650M_splitbygene_nolora_NWT1_train_same_gene_batch_test_same_gene_batch_${COARSE_SELECTION_TYPE}/model.pt \
        --ohe_baseline \
        --eval_regression
    
done

#        --use_lora \
