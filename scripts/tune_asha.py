# tune_lora_asha.py
import os
import re
import subprocess
from collections import deque
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

VAL_AUC_RE = re.compile(r"Val AUC:\s*([0-9]*\.?[0-9]+)")

BASE_DATA_PATH="/gpfs/scratch/jv2807/dms_data"
EMBEDDING_LAYER="layer33_mean"
EVALS_PER_EPOCH = 4
NUM_SAMPLES = 48

# 1) Put your fixed args here (everything except LoRA HPs)
BASE_CMD = [
    "python", "-u", "/gpfs/home/jv2807/dms_contrastive/pipeline.py",
    "--run_name", "PLACEHOLDER",  # overwritten per trial
    "--data_path", BASE_DATA_PATH+"/datasets/all_selection_types.csv",
    "--embeddings_path", BASE_DATA_PATH+"/embeddings/{selection_type}/600M_esmc_mean.h5",
    "--ohe_embeddings_path", BASE_DATA_PATH+"/embeddings/{selection_type}/ohe_embeddings.h5",
    "--model_cache", "/gpfs/scratch/jv2807/cache",
    "--model_name", "esmc",
    "--esm_max_length", "600",
    "--batch_size", "4",
    "--patience", "999",
    "--eval_per_epoch", str(EVALS_PER_EPOCH),
    "--dropout", "0.0",
    "--metadata_path", BASE_DATA_PATH+"/datasets/DMS_substitutions.csv",
    "--num_epochs", "3",
    "--train_same_gene_batch",
    "--test_same_gene_batch",
    "--normalize_to_wt",
    "--split_by_gene",
    "--num_bootstraps", "4",
    "--selection_types", "Stability", "OrganismalFitness", "Binding", "Activity", "Expression",
    "--split_file", "/gpfs/home/jv2807/dms_contrastive/results/all_selection_types_600M_esmc_NWT/data_split.json",
    "--use_lora",
    "--dont_save_model",
    "--subsample", "0.05",
    "--h5_read_only"
]

def trainable(config):
    trial_id = session.get_trial_id()
    run_name = f"asha_lora_{trial_id}"

    cmd = BASE_CMD.copy()
    # replace run_name placeholder
    cmd[cmd.index("PLACEHOLDER")] = run_name

    # append sampled LoRA params
    lora_target_modules = config["lora_target_modules"].split("|")
    hidden_dims = config["hidden_dims"]
    cmd += [
        "--lora_rank", str(config["lora_rank"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--esm_lr", str(config["esm_lr"]),
        "--esm_warmup", str(config["esm_warmup"]),
        "--lora_target_modules", *lora_target_modules,
        "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
        "--dropout", str(config["dropout"]),
        "--learning_rate", str(config["learning_rate"]),
        "--hidden_dims", hidden_dims,
    ]

    env = os.environ.copy()
    # optional: pin one GPU per trial if needed
    # env["CUDA_VISIBLE_DEVICES"] = "0"

    orig_cwd = os.environ.get("TUNE_ORIG_WORKING_DIR", os.getcwd())
    log_dir = os.path.join(orig_cwd, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{trial_id}.log")
    tail = deque(maxlen=80)

    with open(log_path, "w", buffering=1) as log_file:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
        )

        report_iter = 0
        best_auc = float("-inf")

        for line in p.stdout:
            log_file.write(line)
            tail.append(line.rstrip())
            m = VAL_AUC_RE.search(line)
            if m:
                val_auc = float(m.group(1))
                best_auc = max(best_auc, val_auc)
                report_iter += 1
                session.report({
                    "val_auc": val_auc,
                    "best_val_auc": best_auc,
                    "training_iteration": report_iter
                })

        rc = p.wait()

    if rc != 0:
        tail_text = "\n".join(tail)
        raise RuntimeError(
            f"Trial failed with return code {rc}. Pipeline log: {log_path}\nLast output:\n{tail_text}"
        )

    if report_iter == 0:
        raise RuntimeError(f"No Val AUC found in logs; ASHA got no intermediate metrics. Pipeline log: {log_path}")

if __name__ == "__main__":
    search_space = {
        "lora_rank": tune.choice([4, 8, 16, 32]),
        "lora_alpha": tune.choice([8, 16, 32, 64]),
        "esm_lr": tune.loguniform(1e-6, 2e-4),
        "esm_warmup": tune.choice([0.0, 0.01, 0.03, 0.05]),
        "lora_target_modules": tune.choice([
            "layernorm_qkv.1|out_proj",
            "layernorm_qkv.1",
        ]),
        "gradient_accumulation_steps": tune.choice([4, 8, 12]),
        "dropout": tune.choice([0.0, 0.05, 0.1]),
        "learning_rate": tune.loguniform(0.00001, 0.001),
        "hidden_dims": tune.choice([
            "512,256,128",
            "512,256,64",
            "512,256,256",
            "512,256,32",
        ])
    }

    scheduler = ASHAScheduler(
        metric="val_auc",
        mode="max",
        max_t=30,          # max reports (not epochs)
        grace_period=EVALS_PER_EPOCH,    # minimum reports before pruning
        reduction_factor=3
    )

    search_alg = OptunaSearch(metric="val_auc", mode="max")

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 4, "gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=NUM_SAMPLES
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_auc", mode="max")
    print("Best config:", best.config)
    print("Best val_auc:", best.metrics.get("val_auc"))
