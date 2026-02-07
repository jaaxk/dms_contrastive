import pandas as pd

df = pd.read_csv("/gpfs/scratch/jv2807/dms_data/datasets/Stability.csv")

print(pd.Series(df['filename'].value_counts()).describe())