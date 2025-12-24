#!/bin/bash
#SBATCH -p p100
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=62G
#SBATCH --output=logs/%j.out
#SBATCH --job-name=dms_cl

module load python/3.9.7
module load cuda113
source /gpfs/scratch/jvaska/brandes_lab/venv/bin/activate

BASE_DATA_PATH="/gpfs/scratch/jvaska/brandes_lab/dms_data"

COARSE_SELECTION_TYPE="Stability"
EMBEDDING_LAYER="layer33_mean"

echo "Running ${COARSE_SELECTION_TYPE} ${EMBEDDING_LAYER}"

EMBEDDING_PATH="${BASE_DATA_PATH}/embeddings/${COARSE_SELECTION_TYPE}/embeddings_${EMBEDDING_LAYER}.pkl"
RUN_NAME="KMEANS_TEST_${COARSE_SELECTION_TYPE}_${EMBEDDING_LAYER}"

python -u pipeline.py --run_name $RUN_NAME \
    --embeddings_path $EMBEDDING_PATH \
    --data_path $BASE_DATA_PATH/datasets/${COARSE_SELECTION_TYPE}.csv
