#!/bin/bash
#SBATCH -p gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/%j.out
#SBATCH --job-name=dms_cl

module load python/gpu/3.10.6-cuda12.9
source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jv2807/dms_data"

EMBEDDING_LAYER="layer33_mean"

for COARSE_SELECTION_TYPE in "Stability" "OrganismalFitness" "Activity" "Binding" "Expression" ; do
    echo "Running ${COARSE_SELECTION_TYPE} ${EMBEDDING_LAYER}"

    RUN_NAME="650M_NWT_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"

    python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33.h5 \
        --model_cache /gpfs/scratch/jv2807/cache \
        --model_name facebook/esm2_t33_650M_UR50D \
        --esm_max_length 600 \
        --input_dim 1280 \
        --batch_size 32 \
        --patience 4 \
        --dropout 0.0 \
        --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
        --num_epochs 10 \
        --normalize_to_wt
        
        exit 1

done

#         --model_path /gpfs/scratch/jvaska/brandes_lab/results/650M_NORM_SAVE_${COARSE_SELECTION_TYPE}_layer33_mean/projection_head.pt
# "Stability" "OrganismalFitness" "Activity" "Binding" "Expression"
