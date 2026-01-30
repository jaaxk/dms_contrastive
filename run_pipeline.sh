#!/bin/bash
#SBATCH -p v100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/%j.out
#SBATCH --job-name=dms_cl

source venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jvaska/brandes_lab/dms_data"

EMBEDDING_LAYER="layer33_mean"

for COARSE_SELECTION_TYPE in "Binding" ; do
    echo "Running ${COARSE_SELECTION_TYPE} ${EMBEDDING_LAYER}"

    RUN_NAME="650M_OHE+LLR_1_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"

    python -u pipeline.py --run_name $RUN_NAME \
        --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv \
        --embeddings_path $BASE_DATA_PATH/embeddings/${COARSE_SELECTION_TYPE}/650M_t33_mean_layer33.h5 \
        --model_cache /gpfs/scratch/jvaska/cache/esm \
        --model_name facebook/esm2_t33_650M_UR50D \
        --esm_max_length 600 \
        --input_dim 1280 \
        --batch_size 32 \
        --patience 2 \
        --dropout 0.0 \
        --metadata_path $BASE_DATA_PATH/datasets/DMS_substitutions.csv \
        --num_epochs 10 \
        --normalize_to_wt \
        --model_path /gpfs/scratch/jvaska/brandes_lab/results/650M_NORM_SAVE_${COARSE_SELECTION_TYPE}_layer33_mean/projection_head.pt \
        --ohe_baseline

done

#         --model_path /gpfs/scratch/jvaska/brandes_lab/results/650M_NORM_SAVE_${COARSE_SELECTION_TYPE}_layer33_mean/projection_head.pt
# "Stability" "OrganismalFitness" "Activity" "Binding" "Expression"
