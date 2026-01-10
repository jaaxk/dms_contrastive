#!/bin/bash
#SBATCH -p v100
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/%j.out
#SBATCH --job-name=dms_cl

source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jvaska/brandes_lab/dms_data"

EMBEDDING_LAYER="layer33_mean"

for COARSE_SELECTION_TYPE in "Stability" "Binding" "OrganismalFitness" "Expression" "Activity"; do
    echo "Running ${COARSE_SELECTION_TYPE} ${EMBEDDING_LAYER}"

    RUN_NAME="V3_t6_8m_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"

    python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/test_embeddings_esm2_${EMBEDDING_LAYER}_mean.pkl \
        --model_cache /gpfs/scratch/jvaska/cache/esm \
        --model_name facebook/esm2_t33_650M_UR50D \
        --esm_max_length 600 \
        --input_dim 1280 \
        --batch_size 32 \
        --patience 5 \
        --normalize_to_wt \
        --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv

done
