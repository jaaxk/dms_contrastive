import pandas as pd
import json
from pathlib import Path

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
"""
"""
selection_types = ['Stability', 'Binding', 'Expression', 'OrganismalFitness', 'Activity']
results_dir = Path("/gpfs/home/jv2807/dms_contrastive/results")


def avg_metrics(block):
    spearman_values = []
    p_values = []
    for gene_metrics in block.values():
        ridge = gene_metrics.get("ridge", {})
        spearman_key = "spearman_r" if "spearman_r" in ridge else "sperman_r"
        if spearman_key in ridge:
            spearman_values.append(ridge[spearman_key])
        if "p-value" in ridge:
            p_values.append(ridge["p-value"])

    avg_spearman = sum(spearman_values) / len(spearman_values) if spearman_values else float("nan")
    avg_p = sum(p_values) / len(p_values) if p_values else float("nan")
    return avg_spearman, avg_p


print("selection_type\tbaseline_avg_spearman_r\tbaseline_avg_p_value\tprojection_avg_spearman_r\tprojection_avg_p_value")
#for selection in selection_types:
json_path = '/gpfs/home/jv2807/dms_contrastive/results/all_selection_types_LORA_600M_esmc_NWT_eval_spearmanr/result_dicts/ohe_llr_metrics_1.json'#results_dir / f"esmc_spearmanr_{selection}" / "result_dicts" / "ohe_llr_metrics_1.json"
#if not json_path.exists():
#    print(f"{selection}\tMISSING\tMISSING\tMISSING\tMISSING")
#    continue

with open(json_path, 'r') as f:
    result_dict = json.load(f)

baseline_block = result_dict["baseline"]["positional_split"]
projection_block = result_dict["projections"]["positional_split"]

baseline_avg_r, baseline_avg_p = avg_metrics(baseline_block)
projection_avg_r, projection_avg_p = avg_metrics(projection_block)

print(
    f"{'all'}\t{baseline_avg_r:.10f}\t{baseline_avg_p:.12g}\t"
    f"{projection_avg_r:.10f}\t{projection_avg_p:.12g}"
)
"""
"""
Find genes shared across multiple selection types and save overlap details to JSON.
"""
selection_types = ['Stability', 'Binding', 'Expression', 'OrganismalFitness', 'Activity']
data_dir = "/gpfs/scratch/jv2807/dms_data/datasets"

gene_to_selection_types = {}
for selection in selection_types:
    dataset_path = f"{data_dir}/{selection}.csv"
    df = pd.read_csv(dataset_path, usecols=["uniprot_id"])
    unique_genes = df["uniprot_id"].dropna().unique()
    for gene in unique_genes:
        gene_to_selection_types.setdefault(gene, set()).add(selection)

shared_genes = [
    (gene, sorted(list(selections)))
    for gene, selections in gene_to_selection_types.items()
    if len(selections) > 1
]
shared_genes.sort(key=lambda x: (-len(x[1]), x[0]))

print("\nGenes shared across multiple selection types:")
for gene, selections in shared_genes:
    print(f"{gene}\t{len(selections)}\t{selections}")

shared_gene_output = {
    "selection_types": selection_types,
    "data_dir": data_dir,
    "num_shared_genes": len(shared_genes),
    "shared_genes": [
        {
            "gene": gene,
            "num_selection_types": len(selections),
            "selection_types": selections,
        }
        for gene, selections in shared_genes
    ],
}

output_json_path = Path("/gpfs/home/jv2807/dms_contrastive/analysis/shared_genes_across_selection_types.json")
with open(output_json_path, "w") as f:
    json.dump(shared_gene_output, f, indent=2)

print(f"\nSaved shared gene overlap info to: {output_json_path}")
