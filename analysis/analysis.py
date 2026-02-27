import pandas as pd

"""
selection_types = ['Stability', 'Binding', 'Expression', 'OrganismalFitness', 'Activity']
data = {}
for selection in selection_types:
    print(f"\n{selection} dataset:")
    df = pd.read_csv(f"/gpfs/scratch/jv2807/dms_data/datasets/{selection}.csv")
    print(f'Total sequences: {len(df)}')
    print(f'Unique sequences: {len(df["mutated_sequence"].unique())}')
    print(f'Unique genes: {len(df["uniprot_id"].unique())}')
    print(f'Score bin distribution:\n{df["DMS_score_bin"].value_counts()}')
    print(f'Sequence length distribution:\n{df["mutated_sequence"].apply(len).describe()}')
    print(f'Sequences over 600 amino acids: {len(df[df["mutated_sequence"].apply(len) > 600])}')
    print(f'Sequences under 600 amino acids: {len(df[df["mutated_sequence"].apply(len) <= 600])}')

    data[selection] = {'total_len': len(df), 'unique_seqs': len(df['mutated_sequence'].unique()), 'unique_genes': len(df['uniprot_id'].unique()) \
                    , 'bin_high': len(df[df['DMS_score_bin'] == 1]), 'bin_low': len(df[df['DMS_score_bin'] == 0]), \
                    'avg_len': df['mutated_sequence'].apply(len).mean(), 'len_600_plus': len(df[df['mutated_sequence'].apply(len) > 600]), \
                    'len_600_minus': len(df[df['mutated_sequence'].apply(len) <= 600]), 'std_len': df['mutated_sequence'].apply(len).std()}

print()
print(data)
"""
"""
selection_types = ['Stability', 'Binding', 'Expression', 'OrganismalFitness', 'Activity']

data_dir = "/gpfs/scratch/jv2807/dms_data/datasets"
key_cols = ["filename", "uniprot_id", "dms_id"]

records = []
for selection in selection_types:
    path = f"{data_dir}/{selection}.csv"
    df = pd.read_csv(path, usecols=key_cols)
    df["selection"] = selection
    records.append(df)

all_keys = pd.concat(records, ignore_index=True)

for col in key_cols:
    present = all_keys[col].notna()
    shared_values = all_keys.loc[present, ["selection", col]].drop_duplicates()[col].duplicated(keep=False)
    shared_count = int(shared_values.sum())
    print(f"{col}: {shared_count} values shared across multiple selection types")

"""

selection_types = ['Stability', 'Binding', 'Expression', 'OrganismalFitness', 'Activity']
data_dir = "/gpfs/scratch/jv2807/dms_data/datasets"

all_selection_types = pd.concat(
    (
        pd.read_csv(f"{data_dir}/{selection}.csv").assign(coarse_selection_type=selection)
        for selection in selection_types
    ),
    ignore_index=True,
    copy=False,
)
print(all_selection_types.head())
all_selection_types.to_csv(f"{data_dir}/all_selection_types.csv", index=False)
