import os
from pathlib import Path


result_directory = "experiment_results_v2/iteration_1"
dir_path = Path(result_directory)
analysis_file = "analysis_results.txt"

prune_ratios = [10, 30, 50, 70, 90]
centrality_measures = ['BC', 'CC', 'DC', 'EC', 'PR', 'RAND']
preservation_ratios = [90, 95, 97, 99]
p_duration = 3
times = 20
epochs = 20

def generate_analysis():
    accs, senss, specs, f1s = [], [], [], []
    settings = []   # [n][0] = prune_ratio
                    # [n][1] = centrality
                    # [n][2] = preservation_percentile
                    # [n][3] = preservation_duration
                    # [n][4] = times
                    # [n][5] = epochs

    acc_ranking = []

    for i, filename in enumerate(os.listdir(dir_path)):
        experiment_configs = filename.split('_')
        setting = dict()        
        setting['dataset'] = experiment_configs[0]
        setting['prune_ratio'] = experiment_configs[1]
        setting['centrality'] = experiment_configs[2]

        if not setting['centrality'] == "NONE":
            setting['preservation_percentile'] = experiment_configs[3]
            setting['preservation_duration'] = experiment_configs[4]
            setting['times'] = experiment_configs[5]
            setting['epochs'] = experiment_configs[6]
            
        settings.append(setting)

        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r') as file:
            for _ in range(2):
                next(file)
            third_line = file.readline().strip()

            results = third_line.split(',')
            acc, sens, spec, f1 = results[:4]
            accs.append(acc)
            senss.append(sens)
            specs.append(spec)
            f1s.append(f1)

            acc_ranking.append((i, acc))

    acc_ranking.sort(key=lambda x: x[1], reverse=True)
    
    highest = 20
    top = acc_ranking[:highest] 
    print(top)
    for i in range(highest):
        print(settings[top[i][0]])

    return

def save_experiment_results(datalog: dict, epochs):
    dest = Path(datalog['dest'])

    # Create the directory
    if not os.path.exists(dest):
        dest.mkdir(parents=True, exist_ok=True)    
    
    final = datalog['final_results']
    epoch_tests = datalog['epoch_tests']
    epoch_times = datalog['epoch_times']
        
    file_path = dest / "experiment_log.txt"

    with file_path.open('w') as file:
        file.write("Final Results\n")
        file.write(" Accuracy         | Sensitivity      | Specificity      | F1              | AUC \n")
        file.write(f"{epoch_tests[-1][0]},{epoch_tests[-1][1]},{epoch_tests[-1][2]},{epoch_tests[-1][3]},{epoch_tests[-1][4]}\n")
        file.write("Epochs\n")
        file.write(" Time(s)           | Accuracy         | Sensitivity      | Specificity      | F1              | AUC\n")

        for i in range(epochs):
            file.write(f"{epoch_times[i]},{epoch_tests[i][0]},{epoch_tests[i][1]},{epoch_tests[i][2]},{epoch_tests[i][3]},{epoch_tests[-1][4]}\n")
    return
    
def generate_run_script(dataset):
    dest = result_directory
    
    execution_line = "python pytorch_prune_weight_cotrain.py --ratio_graph {0} --ratio_weight {1} --dataset {2} --centrality {3} --preserve_rate {4} --preserve_duration {5} --times {6} --epochs {7} --dest {8}\n"
    none_line = "python pytorch_prune_weight_cotrain.py --ratio_graph {0} --ratio_weight {1} --dataset {2} --centrality {3} --times {4} --epochs {5} --dest {6}\n"
    
    file_name = f'run_experiment_{dataset}.sh'
    current_directory = Path.cwd()
    file_path = current_directory / file_name
    
    
    with file_path.open('w') as file:
        file.write(f"python pytorch_train.py --epoch 0 --dataset {dataset}\n\n")
        for prune_ratio in prune_ratios:
            for centrality in centrality_measures:
                for pratio in preservation_ratios:
                    file.write(execution_line.format(prune_ratio, prune_ratio, dataset, centrality, pratio, p_duration, times, epochs, dest))
                file.write("\n")
            file.write("\n")

        for prune_ratio in prune_ratios:
            file.write(none_line.format(prune_ratio, prune_ratio, dataset, "NONE", times, epochs, dest))

    os.chmod(file_path, 0o755)
    return
