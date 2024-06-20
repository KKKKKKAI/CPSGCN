import json
import argparse
from CPSGCN import CPSGCN
from dataset_manager import *
from model_assessment import ModelAssessor
from model_selection import ModelSelector


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Cora")
parser.add_argument("--model", type=str, default="CPSGCN")
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--result_dir", type=str, default="results")
args = parser.parse_args()

# Create the output results directory if not existed
result_dir = Path(f"{args.result_dir}/{args.model}_{args.dataset}")

if result_dir.exists():
    print("Experiment results existed, continuing experiment.")
else:  
    result_dir.mkdir(parents=True, exist_ok=False)   

all_masks = load_masks(args.dataset)
with open(f'experiment_configs/config_{args.model}.json', 'r') as file:
    loaded_configs = json.load(file)

final_accs = [0] * len(all_masks)
final_sd = [0] * len(all_masks)
final_configs = [None] * len(all_masks)

for i in range(len(all_masks)):
    fold_output_dir = Path(result_dir / f"outer_fold_{i}")
    if not fold_output_dir.exists():
        fold_output_dir.mkdir(parents=True, exist_ok=True)

    test_mask = all_masks[f"outer_fold_{i}"]["test_mask"]

    model_select_dir = Path(fold_output_dir / "model_select")
    if not model_select_dir.exists():
        model_select_dir.mkdir(parents=True, exist_ok=True)
    selector = ModelSelector(len(all_masks[f"outer_fold_{i}"]) - 1, loaded_configs)

    runner_path = Path(f"runner_scripts/{args.model}_{args.dataset}/outer_{i}")
    if not runner_path.exists():
        runner_path.mkdir(parents=True, exist_ok=True)

    for j in range(len(all_masks[f"outer_fold_{i}"]) - 1):
        data_mask = (i, j)
        inner_runner_path = runner_path
        
        selector.select_model(data_mask, loaded_configs, args.dataset, model_select_dir, inner_runner_path)
    
    best_acc, best_config = selector.get_best_model()
    print("best config: ", best_config)
    print("best accuracy: ", best_acc)
    
    model_assess_dir = Path(fold_output_dir / "model_assess")
    if not model_assess_dir.exists():
        model_assess_dir.mkdir(parents=True, exist_ok=True)
    assess_runner_path = runner_path / "assess.sh"
    assessor = ModelAssessor(args.repeat, best_config, args.dataset, model_assess_dir)
    assessor.assess(data_mask, assess_runner_path)

    final_accs[i] = assessor.get_avg_acc()
    final_sd[i] = assessor.get_sd()
    final_configs[i] = best_config

    config_and_result = {"configs": best_config, "outer_test_acc": final_accs[i], "outer_test_sd": final_sd[i]}
    output_file = fold_output_dir / 'outer_fold_best_config_and_result.json'
    with open(output_file, 'w') as file:
        json.dump(config_and_result, file)

max_acc = max(final_accs)
max_idx = final_accs.index(max_acc)
overall_best_config = final_configs[0]
sd_max = final_sd[max_idx]

configs_and_result = {"configs": overall_best_config, "final_acc": max_acc, "final_sd": sd_max}

print(f"final accuracy: {max_acc}, standard deviation: {sd_max}, \n best config: {overall_best_config}")

os.makedirs(result_dir, exist_ok=True)

with open(f'{result_dir}/overall_best_config_{args.model[0]}.json', 'w') as file:
    json.dump(configs_and_result, file)

