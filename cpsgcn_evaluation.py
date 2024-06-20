import os
import csv
import numpy as np

prune_ratios = [10, 30, 50, 70, 90]
preserve_rates = [90, 95, 97, 99]
centralities = ["BC", "DC", "CC", "EC", "PR", "RAND"]

def generate_evaluation():
    with open("runner_scripts/CiteSeer_CPSGCN.sh", "w") as script_file:
        for prune_ratio in prune_ratios:
            for preserve_rate in preserve_rates:
                for centrality in centralities:
                    line = (
                        f"python CPSGCN.py --use_gpu --prune_ratio {prune_ratio} "
                        f"--dataset_name CiteSeer --centrality {centrality} "
                        f"--preserve_rate {preserve_rate} "
                        f"--dest results/CPSGCN_CiteSeer_Table/{prune_ratio}_{centrality}_{preserve_rate}\n"
                    )
                    script_file.write(line)
                script_file.write("\n")
            script_file.write("\n")

def read_experiment_logs(base_directory):
    results = {}

    # Iterate through each iteration directory
    for iteration in range(1, 6):
        iteration_dir = os.path.join(base_directory, f'iteration_{iteration}')
        if not os.path.isdir(iteration_dir):
            continue

        # Initialize the iteration in the results dictionary
        results[iteration] = {}

        # Iterate through each configuration directory
        for prune_ratio in [10, 30, 50, 70, 90]:
            for centrality in ['BC', 'CC', 'DC', 'EC', 'PR', 'RAND']:
                for preserve_ratio in [90, 95, 97, 99]:
                    config_dir = f'{prune_ratio}_{centrality}_{preserve_ratio}'
                    config_path = os.path.join(iteration_dir, config_dir)

                    if not os.path.isdir(config_path):
                        continue

                    log_file_path = os.path.join(config_path, 'experiment_log.txt')
                    if not os.path.isfile(log_file_path):
                        continue

                    # Initialize the configuration in the results dictionary
                    if config_dir not in results[iteration]:
                        results[iteration][config_dir] = {}

                    # Read the experiment_log.txt file
                    with open(log_file_path, 'r') as log_file:
                        lines = log_file.readlines()
                        if len(lines) >= 3:
                            metrics_line = lines[2].strip()
                            metrics = metrics_line.split(',')

                            # Parse the metrics
                            if len(metrics) == 5:
                                acc, sens, spec, f1, auc = map(float, metrics)
                                results[iteration][config_dir] = {
                                    'acc': acc,
                                    'sens': sens,
                                    'spec': spec,
                                    'f1': f1,
                                    'auc': auc
                                }

    return results

def write_results_to_file(results, output_file):
    with open(output_file, 'w') as f:
        for iteration in range(1, 6):
            for centrality in ['BC', 'CC', 'DC', 'EC', 'PR', 'RAND']:
                for preserve_ratio in [90, 95, 97, 99]:
                    row_data = []
                    for metric in ['acc', 'sens', 'spec', 'f1', 'auc']:
                        for prune_ratio in [10, 30, 50, 70, 90]:
                            config_dir = f'{prune_ratio}_{centrality}_{preserve_ratio}'
                            if config_dir in results[iteration]:
                                row_data.append(str(results[iteration][config_dir].get(metric, '')))
                            else:
                                row_data.append('')
                    f.write('\t'.join(row_data) + '\n')
            f.write('\n')

if __name__ == "__main__":
    # generate_evaluation()
    
    base_directory = 'results/CPSGCN_CiteSeer_Table/'
    experiment_results = read_experiment_logs(base_directory)
    write_results_to_file(experiment_results, 'cpsgcn_cora_result_table.txt')