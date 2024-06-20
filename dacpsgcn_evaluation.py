import os
import json

from pathlib import Path
from GACPSGCN import DACPSGCN
from runner_script_manager import *

def generate_evaluation():
    # Create the output results directory if not existed
    result_dir = Path(f"results/DACPSGCN_evaluation")

    if result_dir.exists():
        print("Experiment results existed, continuing experiment.")
    else:  
        result_dir.mkdir(parents=True, exist_ok=False)   

    configs = {
        "model": "DACPSGCN",
        "centrality": "BC_EC",
        "use_gpu": True,
        "adj_lr": 0.001,
        "w_lr": 0.02,
        "prune_ratio": 50,
        "preserve_rate": 97,
        "dataset_name": "Cora"
    }

    # Cora dataset
    BC_sf = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    EC_sf = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

    # CiteSeer dataset 
    # BC_sf = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    # EC_sf = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

    runner_path = Path("runner_scripts/DACPSGCN_eval.sh")

    configs["model"] = "ACPSGCN"
    ACPSGCN_zscore = result_dir / "ACPSGCN_zscore"
    if not ACPSGCN_zscore.exists():
        ACPSGCN_zscore.mkdir(parents=True, exist_ok=True)
    
    with open(runner_path, 'a') as file:
        file.write(f"python ACPSGCN.py --use_gpu --dest {ACPSGCN_zscore} --dataset_name Cora --w_lr 0.02 --adj_lr 0.001 --prune_ratio 50 --preserve_rate 97 --centrality BC_EC --ac_select zscore\n")

    with open(ACPSGCN_zscore / "config", 'w') as file:
        configs["ac_select"] = "zscore"
        json.dump(configs, file)


    ACPSGCN_minmax = result_dir / "ACPSGCN_minmax" 
    if not ACPSGCN_minmax.exists():
        ACPSGCN_minmax.mkdir(parents=True, exist_ok=True)
    with open(ACPSGCN_minmax / "config", 'w') as file:
        configs["ac_select"] = "minmax"
        json.dump(configs, file)

    with open(runner_path, 'a') as file:
        file.write(f"python ACPSGCN.py --use_gpu --dest {ACPSGCN_minmax} --dataset_name Cora --w_lr 0.02 --adj_lr 0.001 --prune_ratio 50 --preserve_rate 97 --centrality BC_EC --ac_select minmax\n")

    configs["model"] = "DACPSGCN"
    DACPSGCN_dir = result_dir / "DACPSGCN_results"
    if not DACPSGCN_dir.exists():
        DACPSGCN_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    for BC in BC_sf:
        for DC in EC_sf:
            configs["BC_sf"] = BC
            configs["EC_sf"] = DC
            DACPSGCN_config_dir = DACPSGCN_dir / f"config_{i}"
            if not DACPSGCN_config_dir.exists():
                DACPSGCN_config_dir.mkdir(parents=True, exist_ok=True)

            with open(DACPSGCN_config_dir/"config.json", 'w') as file:
                json.dump(configs, file)
            print(BC, DC)
            generate_runner(configs, runner_path, DACPSGCN_config_dir)
            i += 1

    os.chmod(runner_path, 0o755)
    # execute_shell_script(runner_path)

def get_best():
    result_dir = Path(f"results/DACPSGCN_evaluation/DACPSGCN_results")

    accs = [0] * 81
    
    directories = sorted([d.name for d in result_dir.iterdir() if d.is_dir()])
    for j in range(len(directories)):
        with open(result_dir / directories[j] / "experiment_log.txt", 'r') as file:
            for _ in range(2):
                next(file)
            third_line = file.readline().strip()
            results = third_line.split(',')
            acc, _, _, _ = results[:4]
            accs[j] = float(acc)

    max_acc = max(accs)
    max_index = accs.index(max_acc)
    print(max_acc)
    print(max_index)

    with open(f'{result_dir / directories[max_index]}/config.json', 'r') as file:
        configs = json.load(file)

    with open(result_dir / "best_config_and_result.json", 'w') as file:
        json.dump({"config":configs, "acc": max_acc}, file)


if __name__ == "__main__":
    # generate_evaluation()
    get_best()