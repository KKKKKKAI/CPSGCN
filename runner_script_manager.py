import subprocess

def generate_runner(configs, runner_path, output_path):
    if configs["model"] == "CPSGCN" or configs["model"] == "ACPSGCN":
        use_gpu = "--use_gpu" if configs["use_gpu"] == True else None
        total_epochs = configs["epochs"]
        dest = output_path
        dataset_name = configs["dataset_name"]
        w_lr = configs["w_lr"]
        adj_lr = configs["adj_lr"]
        prune_ratio = configs["prune_ratio"]
        preserve_rate = configs["preserve_rate"]
        centrality = configs["centrality"]
        outer_i, inner_j = configs["data_mask"] 
        test_run = ""
        if "test_run" in configs:
            test_run = "--test_run"
        
        if configs["model"] == "CPSGCN":
            execution_line = "python CPSGCN.py {0} --total_epochs {1} --dest {2} --dataset_name {3} --w_lr {4} --adj_lr {5} --prune_ratio {6} --preserve_rate {7} --centrality {8} --outer_k {9} --inner_k {10} {11}\n"
            execution_line = execution_line.format(use_gpu, total_epochs, dest, dataset_name, w_lr, adj_lr, prune_ratio, preserve_rate, centrality, outer_i, inner_j, test_run)
        else:
            ac_select = configs["ac_select"]
            execution_line = "python ACPSGCN.py {0} --total_epochs {1} --dest {2} --dataset_name {3} --w_lr {4} --adj_lr {5} --prune_ratio {6} --preserve_rate {7} --centrality {8} --outer_k {9} --inner_k {10} --ac_select {11} {12}\n"
            execution_line = execution_line.format(use_gpu, total_epochs, dest, dataset_name, w_lr, adj_lr, prune_ratio, preserve_rate, centrality, outer_i, inner_j, ac_select, test_run)
        
    elif configs["model"] == "SSP":
        total_epochs = configs["epochs"]
        dest = output_path
        dataset_name = configs["dataset_name"]
        lr = configs["lr"]
        hidden = configs["hidden"]
        dropout = configs["dropout"]
        optimizer = configs["optimizer"]
        outer_i, inner_j = configs["data_mask"] 

        preconditioner = ""
        if configs["preconditioner"] == "KFAC":
            preconditioner = f"--preconditioner {configs['preconditioner']}"
        hyperparam = ""
        if not configs["hyperparam"] == "":
            hyperparam = f"--hyperparam {configs['hyperparam']}"
        gamma = ""
        if "gamma" in configs:
            gamma = f" --gamma {configs['gamma']}"
        eps = ""
        if "eps" in configs:
            eps = f"--eps {configs['eps']}"
        update_freq = ""
        if "update_freq" in configs:
            update_freq = f"--update_freq {configs['update_freq']}"
        test_run = ""
        if "test_run" in configs:
            test_run = "--test_run"
        
        execution_line = "python benchmark_models/ssp/gcn.py --epochs {0} --dest {1} --dataset {2} --lr {3} --hidden {4} --dropout {5} --optimizer {6} {7} {8} {9} {10} {11} --outer_k {12} --inner_k {13} {14}\n"
        execution_line = execution_line.format(total_epochs, dest, dataset_name, lr, hidden, dropout, optimizer, preconditioner, hyperparam, gamma, eps, update_freq, outer_i, inner_j, test_run)
    elif configs["model"] == "SGCN":
        use_gpu = "--use_gpu" if configs["use_gpu"] == True else None
        total_epochs = configs["epochs"]
        dest = output_path
        dataset_name = configs["dataset_name"]
        ratio_weight = configs["ratio_weight"]
        ratio_graph = configs["ratio_graph"]
        w_lr = configs["w_lr"]
        adj_lr = configs["adj_lr"]
        outer_i, inner_j = configs["data_mask"] 

        test_run = ""
        if "test_run" in configs:
            test_run = "--test_run"
        
        execution_line = "python benchmark_models/sgcn/sgcn.py {0} --epochs {1} --dest {2} --dataset {3} --ratio_weight {4} --ratio_graph {5} --w_lr {6} --adj_lr {7} --outer_k {8} --inner_k {9} {10}\n"
        execution_line = execution_line.format(use_gpu, total_epochs, dest, dataset_name, ratio_weight, ratio_graph, w_lr, adj_lr, outer_i, inner_j, test_run)
    elif configs["model"] == "GraphSAGE":
        use_gpu = "--use_gpu" if configs["use_gpu"] == True else None
        total_epochs = configs["epochs"]
        dest = output_path
        dataset_name = configs["dataset_name"]
        hidden_units = configs["hidden_units"]
        lr = configs["lr"]
        num_layers = configs["num_layers"]
        aggregation = configs["aggregation"]
        outer_i, inner_j = configs["data_mask"] 

        test_run = ""
        if "test_run" in configs:
            test_run = "--test_run"
        
        execution_line = "python benchmark_models/GraphSAGE/GraphSAGE.py {0} --total_epochs {1} --dest {2} --dataset {3} --hidden_units {4} --lr {5} --num_layers {6} --aggregation {7} --outer_k {8} --inner_k {9} {10}\n"
        execution_line = execution_line.format(use_gpu, total_epochs, dest, dataset_name, hidden_units, lr, num_layers, aggregation, outer_i, inner_j, test_run)
    
    elif configs["model"] == "DACPSGCN":
        use_gpu = "--use_gpu" if configs["use_gpu"] == True else None
        dest = output_path
        dataset_name = configs["dataset_name"]
        prune_ratio = configs["prune_ratio"]
        preserve_rate = configs["preserve_rate"]
        centrality = configs["centrality"]
        w_lr = configs["w_lr"]
        adj_lr = configs["adj_lr"]

        
        BC_sf = ""
        if "BC_sf" in configs:
            BC_sf = f"--BC_sf {configs['BC_sf']}"
        CC_sf = ""
        if "CC_sf" in configs:
            CC_sf = f"--CC_sf {configs['CC_sf']}"
        DC_sf = ""
        if "DC_sf" in configs:
            DC_sf = f"--DC_sf {configs['DC_sf']}"
        EC_sf = ""
        if "EC_sf" in configs:
            EC_sf = f"--EC_sf {configs['EC_sf']}"
            
        execution_line = "python DACPSGCN.py {0} --dest {1} --dataset_name {2} --w_lr {3} --adj_lr {4} --prune_ratio {5} --preserve_rate {6} --centrality {7} {8} {9} {10} {11}\n"
        execution_line = execution_line.format(use_gpu, dest, dataset_name, w_lr, adj_lr, prune_ratio, preserve_rate, centrality, BC_sf, CC_sf, DC_sf, EC_sf)


    with open(runner_path, 'a') as file:
        file.write(execution_line)

def execute_shell_script(script_path):
    try:
        # Use Popen to start the script
        process = subprocess.Popen(['sh', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Communicate with the process to capture output
        stdout, stderr = process.communicate()
        print(f"Output of {script_path}:\n{stdout}")
        if stderr:
            print(f"Errors:\n{stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {e}")
        return None

