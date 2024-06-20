import os
import json

from pathlib import Path
from itertools import product

from runner_script_manager import *

class ModelSelector:
    def __init__(self, inner_folds, configs):
        self.inner_fold = inner_folds

        self.inner_counter = 0

        values = list(configs.values())
        keys = list(configs.keys())

        combinations = product(*values)

        self.configs = [dict(zip(keys, combination)) for combination in combinations]
        self.accs = [[0] * self.inner_fold for _ in range(len(self.configs))]
        
        self.best_config = {}

    def _gets_best(self, accs, iterations, output_dir):
        # get average accuracy for each config innerfold

        for i in range(iterations):
            inner_fold_output_dir = Path(output_dir / f"inner_{i}/")
            directories = sorted([d.name for d in inner_fold_output_dir.iterdir() if d.is_dir()])
            for j in range(len(directories)):
                with open(inner_fold_output_dir / directories[j] / "experiment_log.txt", 'r') as file:
                    for _ in range(2):
                        next(file)
                    third_line = file.readline().strip()
                    results = third_line.split(',')
                    acc, _, _, _ = results[:4]
                    accs[j][i] = float(acc)

        avg_accs = [0] * len(directories)
        denom = len(accs[0])
        for j in range(len(directories)):
            avg_accs[j] = sum(accs[j]) / denom

        # select best config
        max_acc = max(avg_accs)
        max_index = avg_accs.index(max_acc)
        print(max_acc)
        print(max_index)
        with open(f'{inner_fold_output_dir / directories[max_index]}/config.json', 'r') as file:
            configs = json.load(file)

        return max_acc, configs, max_index

    def _dump_configs(self, output_dir, config, best=False):
        if best:
            output_file = output_dir / 'best_config.json'
        else:
            output_file = output_dir / 'config.json'
        with open(output_file, 'w') as file:
            json.dump(config, file)

    def select_model(self, data_mask, loaded_configs, dataset_name, output_dir=Path("results"), runner_path=Path("runner_scipts/")):
        tmp_config = {"model": loaded_configs["model"][0],
                        "use_gpu": loaded_configs["use_gpu"][0],
                        "epochs": loaded_configs["epochs"][0],
                        "dataset_name": dataset_name,
                        "data_mask": data_mask} 

        if loaded_configs["model"][0] == "CPSGCN" or loaded_configs["model"][0] == "ACPSGCN":     
            if loaded_configs["model"][0] == "CPSGCN":
                centrality_configs = dict((k, loaded_configs[k]) for k in ('centrality', 'prune_ratio', 'preserve_rate'))
            else:
                centrality_configs = dict((k, loaded_configs[k]) for k in ('centrality', 'prune_ratio', 'preserve_rate', 'ac_select'))

            centrality_combinations = product(*centrality_configs.values())
            centrality_configs = [dict(zip(centrality_configs.keys(), combination)) for combination in centrality_combinations]
            centrality_path = runner_path / f"inner_{self.inner_counter}.sh"

            script_generated = False
            if os.path.exists(centrality_path):
                script_generated = True

            for i, centrality_config in enumerate(centrality_configs):
                print("centrality selection:", centrality_config)
                
                centrality_output_dir = Path(output_dir / f"inner_{self.inner_counter}" / f"config_{i}")
                if not centrality_output_dir.exists():
                    centrality_output_dir.mkdir(parents=True, exist_ok=True)
                
                # default values for adj_lr and w_lr for centrality configs selection
                tmp_config["adj_lr"] = 0.001
                tmp_config["w_lr"] = 0.03

                self._dump_configs(centrality_output_dir, {**tmp_config, **centrality_config})
                
                if not script_generated:
                    generate_runner({**tmp_config, **centrality_config}, centrality_path, centrality_output_dir)
            
            os.chmod(centrality_path, 0o755)
            execute_shell_script(centrality_path)
            
            self.inner_counter += 1

            if self.inner_counter == self.inner_fold:
                _, best_cent_config, _ = self._gets_best(self.accs, self.inner_fold, output_dir)
                print(best_cent_config)
                
                net_output_dir = Path(output_dir / f"best_config" / "net_configs")
                if not net_output_dir.exists():
                    net_output_dir.mkdir(parents=True, exist_ok=True)

                net_configs = dict((k, loaded_configs[k]) for k in ('adj_lr', 'w_lr'))
                net_combinations = product(*net_configs.values())
                net_configs = [dict(zip(net_configs.keys(), combination)) for combination in net_combinations]

                accs = [[0] * 3 for _ in range(len(net_configs))] # take 3 times avg for each net config

                net_path = runner_path / "net.sh" 
                script_generated = False
                if os.path.exists(net_path):
                    script_generated = True

                for i in range(3): # take 3 times avg for each net config
                    for j, net_config in enumerate(net_configs):
                        print("net selection:", net_config)
                        
                        inner_net_output_dir = Path(net_output_dir / f"inner_{i}" / f"config_{j}")
                        if not inner_net_output_dir.exists():
                            inner_net_output_dir.mkdir(parents=True, exist_ok=True)

                        tmp_config["adj_lr"] = net_config["adj_lr"]
                        tmp_config["w_lr"] = net_config["w_lr"]

                        self._dump_configs(inner_net_output_dir, {**tmp_config, **best_cent_config})

                        if not script_generated:
                            generate_runner({**tmp_config, **best_cent_config}, net_path, inner_net_output_dir)
                        
                
                del tmp_config["adj_lr"] 
                del tmp_config["w_lr"] 
                # check net config selection

                os.chmod(net_path, 0o755)
                execute_shell_script(net_path)

                
                best_net_acc, best_net_config, _ = self._gets_best(accs, 3, net_output_dir)

                self.best_config = {"config": {**tmp_config, **best_net_config, **best_cent_config}, "acc": best_net_acc}

        elif loaded_configs["model"][0] == "SSP":
            eps = [-3, 0, 10]                                            
            update_freq = [4, 8, 16, 32, 64, 128]                        
            gamma = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 ]

            ssp_path = runner_path / f"inner_{self.inner_counter}.sh"

            script_generated = False
            if os.path.exists(ssp_path):
                script_generated = True

            i = 0
            for config in self.configs:
                if config["preconditioner"] == '':
                    if config["hyperparam"] == 'eps' or config["hyperparam"] == 'update_freq':
                        continue
                elif config["preconditioner"] == 'KFAC':
                    if config["hyperparam"] == '':
                        continue
                    elif config["hyperparam"] == 'eps':
                        continue
                
                if config["hyperparam"] == 'eps':
                    for ep in eps:
                        config["eps"] = ep
                        self.ssp_selection(tmp_config, config, i, output_dir, ssp_path, script_generated)
                        i += 1

                elif config["hyperparam"] == 'update_freq':
                    for uf in update_freq:
                        config["update_freq"] = uf
                        self.ssp_selection(tmp_config, config, i, output_dir, ssp_path, script_generated)
                        i += 1

                elif config["hyperparam"] == 'gamma':
                    for g in gamma:
                        config["gamma"] = g
                        self.ssp_selection(tmp_config, config, i, output_dir, ssp_path, script_generated)
                        i += 1

                elif config["hyperparam"] == '':
                    self.ssp_selection(tmp_config, config, i, output_dir, ssp_path, script_generated)
                    i += 1
            
            os.chmod(ssp_path, 0o755)
            execute_shell_script(ssp_path)
            self.inner_counter += 1
            if self.inner_counter == self.inner_fold:
                accs = [[0] * self.inner_fold for _ in range(i)] # take 3 times avg for each net config

                best_acc, best_ssp_config, _ = self._gets_best(accs, self.inner_fold, output_dir)

                
                self.best_config = {"config": {**tmp_config, **best_ssp_config}, "acc": best_acc}

        elif loaded_configs["model"][0] == "SGCN":
            sgcn_path = runner_path / f"inner_{self.inner_counter}.sh"

            script_generated = False
            if os.path.exists(sgcn_path):
                script_generated = True

            for i, config in enumerate(self.configs):
                print("sgcn selection:", config)
                
                sgcn_output_dir = Path(output_dir / f"inner_{self.inner_counter}" / f"config_{i}")

                if not sgcn_output_dir.exists():
                    sgcn_output_dir.mkdir(parents=True, exist_ok=True)

                self._dump_configs(sgcn_output_dir, {**tmp_config, **config})
                
                if not script_generated:
                    generate_runner({**tmp_config, **config}, sgcn_path, sgcn_output_dir)
                
            os.chmod(sgcn_path, 0o755)
            execute_shell_script(ssp_path)
            self.inner_counter += 1
            
            if self.inner_counter == self.inner_fold:
                best_acc, best_sgc_config, _ = self._gets_best(self.accs, self.inner_fold, output_dir)
                
                self.best_config = {"config": {**tmp_config, **best_sgc_config}, "acc": best_acc}

        elif loaded_configs["model"][0] == "GraphSAGE":
            graphsage_path = runner_path / f"inner_{self.inner_counter}.sh"

            script_generated = False
            if os.path.exists(graphsage_path):
                script_generated = True

            for i, config in enumerate(self.configs):
                print("graphsage selection:", config)
                
                graphsage_output_dir = Path(output_dir / f"inner_{self.inner_counter}" / f"config_{i}")

                if not graphsage_output_dir.exists():
                    graphsage_output_dir.mkdir(parents=True, exist_ok=True)

                self._dump_configs(graphsage_output_dir, {**tmp_config, **config})
                
                if not script_generated:
                    generate_runner({**tmp_config, **config}, graphsage_path, graphsage_output_dir)
                
            os.chmod(graphsage_path, 0o755)
            execute_shell_script(graphsage_path)
            self.inner_counter += 1
            
            if self.inner_counter == self.inner_fold:
                best_acc, best_sgc_config, _ = self._gets_best(self.accs, self.inner_fold, output_dir)
                
                self.best_config = {"config": {**tmp_config, **best_sgc_config}, "acc": best_acc}


        # save best config
        self._dump_configs(output_dir, self.best_config, best=True)

    def get_best_model(self):
        return self.best_config["acc"], self.best_config["config"]
    
    def ssp_selection(self, tmp_config, config, i, output_dir, ssp_path, script_generated=False):    
        ssp_output_dir = Path(output_dir / f"inner_{self.inner_counter}" / f"config_{i}")

        print(config)
                
        if not ssp_output_dir.exists():
            ssp_output_dir.mkdir(parents=True, exist_ok=True)

        self._dump_configs(ssp_output_dir, {**tmp_config, **config})
        
        if not script_generated:
            generate_runner({**tmp_config, **config}, ssp_path, ssp_output_dir)
       
        
    
