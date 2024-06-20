import os
import torch
import argparse
import numpy as np
import torch_geometric.transforms as T

from pathlib import Path
from torch_geometric.datasets import Planetoid

test_ratio = 0.2
val_ratio = 0.2
train_ratio = 0.6

def generate_masks(dataset_name, m_outer, n_inner, root='data/Planetoid/'):
    # Fetch the dataset
    dataset = Planetoid(root=root+dataset_name, name=dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)

    # Directory to save the masks
    save_dir = f'data_split/{dataset_name}' 
    if not Path(save_dir).exists():
        os.makedirs(save_dir, exist_ok=True)
    else:
        print("data split already generated, delete existing split to generate new version")
        return
    
    all_masks = {}

    # Split indices for outer folds
    # outer_fold_size = min(num_nodes // m_outer, int(num_nodes * test_ratio)) # REMEBER To Remove min once finished
    outer_fold_size = num_nodes // m_outer 

    for o in range(m_outer):
        all_masks[f'outer_fold_{o}'] = {}
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        outer_test_indices = indices[o * outer_fold_size:(o + 1) * outer_fold_size]
        test_mask[outer_test_indices] = True
        all_masks[f'outer_fold_{o}']['test_mask'] = test_mask

        outer_train_indices = np.concatenate((indices[:o * outer_fold_size], indices[(o + 1) * outer_fold_size:]))
        
        for i in range(n_inner):
            all_masks[f'outer_fold_{o}'][f'select_mask_{i}'] = {}
            # Split each outer fold into inner folds
            # inner_fold_size = min(len(outer_train_indices) // n_inner, int(num_nodes * val_ratio)) # REMEBER To Remove min once finished
            inner_fold_size = len(outer_train_indices) // n_inner

            inner_train_indices = np.concatenate((outer_train_indices[:i * inner_fold_size], outer_train_indices[(i + 1) * inner_fold_size:])) 
            inner_val_indices = outer_train_indices[i * inner_fold_size:(i + 1) * inner_fold_size]

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[inner_train_indices] = True    
            val_mask[inner_val_indices] = True

            all_masks[f'outer_fold_{o}'][f'select_mask_{i}']['train_mask'] = train_mask 
            all_masks[f'outer_fold_{o}'][f'select_mask_{i}']['val_mask'] = val_mask

    # Save the all_masks dictionary
    mask_path = os.path.join(save_dir, 'all_masks.pt')
    torch.save(all_masks, mask_path)

def load_masks(dataset_name):
    save_dir = f'data_split/{dataset_name}'
    mask_path = os.path.join(save_dir, 'all_masks.pt')
    all_masks = torch.load(mask_path)
    return all_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--outer_k', type=int, default=5)
    parser.add_argument('--inner_k', type=int, default=2)
    args = parser.parse_args()

    generate_masks(args.dataset, args.outer_k, args.inner_k)
    
    # Example of how to load the masks
    masks = load_masks(args.dataset)
    print(masks)
