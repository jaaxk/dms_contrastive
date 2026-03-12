#!/bin/bash
#SBATCH -p gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --output=logs/%j.out
#SBATCH --job-name=dms_cl_sp
#SBATCH --signal=B:TERM@60

SINGULARITY="singularity exec --nv --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

BASE_DATA_PATH="/scratch/jv2807/dms_data"

EMBEDDING_LAYER="layer33_mean"

for COARSE_SELECTION_TYPE in "Binding" "OrganismalFitness" ; do
    echo "Running ${COARSE_SELECTION_TYPE} ${EMBEDDING_LAYER}"

    RUN_NAME="esmc_spearmanr_${COARSE_SELECTION_TYPE}"

    $SINGULARITY /bin/bash -c "source /ext3/env.sh; cd /home/jv2807/dms_contrastive && python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/{selection_type}/600M_esmc_mean.h5 \
        --ohe_embeddings_path $BASE_DATA_PATH/embeddings/{selection_type}/ohe_embeddings.h5 \
        --model_cache /scratch/jv2807/cache \
        --model_name esmc \
        --esm_max_length 600 \
        --input_dim 1152 \
        --batch_size 64 \
        --gradient_accumulation_steps 8 \
        --patience 2 \
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
        --split_file /home/jv2807/dms_contrastive/results/650M_splitbygene_lora2_${COARSE_SELECTION_TYPE}_layer33_mean/data_split.json \
        --ohe_baseline \
        --num_bootstraps 1 \
        --eval_regression \
        --selection_types ${COARSE_SELECTION_TYPE} \
        --model_path /home/jv2807/dms_contrastive/results/600M_esmc_NWT_${COARSE_SELECTION_TYPE}/model.pt"

done

#         --model_path /home/jv2807/dms_contrastive/results/600M_esmc_NWT_${COARSE_SELECTION_TYPE}/model.pt \


#        --use_lora \
#--model_name facebook/esm2_t33_650M_UR50D \
#--embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33.h5 \
#        --use_lora \

#"Stability" "OrganismalFitness" "Binding" "Activity" "Expression"

echo "Hello, World!"