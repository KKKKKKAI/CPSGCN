import os 
import json

dir = 'experiment_configs'
os.makedirs(dir, exist_ok=True)

models = ['CPSGCN']
configs = {}

configs["CPSGCN"] = {
    "model": ['CPSGCN'],
    "use_gpu": [True],
    "epochs": [400],
    "adj_lr": [0.001, 0.0001],
    "w_lr": [0.01, 0.02, 0.03], # consider only have train for nets after centrality configs 
    "centrality": ['RAND', 'BC', 'DC', 'PR', 'EC', 'CC'], # consider only sensible combinations of centrality measures
    "prune_ratio": [10, 30, 50, 70, 90],
    "preserve_rate": [90, 95, 97, 99]
}

configs["ACPSGCN"] = {
    "model": ['ACPSGCN'],
    "use_gpu": [True],
    "epochs": [400],
    "adj_lr": [0.001],
    "w_lr": [0.03], # consider only have train for nets after centrality configs 
    "centrality": ['BC_DC', 'BC_CC', 'BC_EC', 'DC_CC', 'DC_EC', 'CC_EC',
                    'BC_DC_CC', 'BC_DC_EC', 'BC_EC_CC', 'DC_CC_EC',
                    'BC_DC_CC_EC'], # consider only sensible combinations of centrality measures
    "prune_ratio": [10, 30, 50, 70, 90],
    "preserve_rate": [90, 95, 97, 99],
    "ac_select": ["minmax", "zscore"]
}

configs["SSP"] = {
    "model": ['SSP'],
    "use_gpu": [True],
    "epochs": [400],
    "preconditioner": ['KFAC',  ''],
    "optimizer": ['SGD', 'Adam'], 
    "hyperparam": ['gamma', 'eps', 'update_freq', ''], 
    "lr": [0.01, 0.001],
    "hidden": [16, 32],
    "dropout": [0.0, 0.05, 0.5]
}

configs["SGCN"] = {
    "model": ['SGCN'],
    "use_gpu": [True],
    "epochs": [400],
    "ratio_weight": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    "ratio_graph": [10, 20, 30, 40, 50, 60, 70, 80, 90], 
    "adj_lr": [0.001, 0.0001],
    "w_lr": [0.01, 0.02, 0.03]
}

configs["GraphSAGE"] = {
    "model": ['GraphSAGE'],
    "use_gpu": [True],
    "epochs": [400],
    "num_layers": [3, 5],
    "hidden_units": [32, 64],
    "lr": [0.01, 0.001, 0.0001],    
    "aggregation": ['mean', 'max', 'sum']
}

# Save the dictionary to a file
for config in configs.values():
    print(config)
    with open(f'{dir}/config_{config["model"][0]}.json', 'w') as file:
        json.dump(config, file)

