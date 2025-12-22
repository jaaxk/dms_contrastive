## Data
- dms_data/embeddings: [https://drive.google.com/drive/folders/191eLieIGhqbDIRTeUKvrNBeo4SYrId0a?usp=share_link]
- dms_data/datasets: [https://drive.google.com/drive/folders/1kHTNgm-uNTQlvTQq-SMQKgnpv2RPfH_J?usp=sharing]

## Environment setup
- Using Python 3.9.16
- python -m venv venv
- pip install -r requirements.txt
- change torch CUDA version depending on system


Personal setup notes:
- Running on p100 node
1. module load python/3.9.7
2. module load cuda113
3. source venv/bin/activate
4. ./run_pipeline.sh
