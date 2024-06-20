import time
import torch
import argparse
import numpy as np
import dataset_manager
import torch_geometric.transforms as T

from ebgcn_net import Net
from net_utils import *
from torch.nn import Parameter, ParameterList
from torch_geometric.datasets import Planetoid
from net_utils import acp_prune_adj, test, update_gradients_adj
from CPSGCN import CPSGCN

class DACPSGCN(CPSGCN):
    def __init__(self, 
				use_gpu=False, 
				total_epochs=400, 
				dest=None, 
				dataset_name='Cora',
				w_lr=0.03, 
				adj_lr=0.001,
				prune_ratio=10, # prune below X percentile of the graph
				preservation_duration=3, 
				preserve_rate=90, 
				centrality='RAND',
                ac_select='minmax'):
        super().__init__(use_gpu, total_epochs, dest, dataset_name, w_lr, adj_lr, 
                        prune_ratio, preservation_duration, preserve_rate, centrality)
        
        self.ac_select = ac_select

    def set_up(self, scale_factors):
        # Initialising node preservation 
        node_occasions = DynamicCPNodeCount()
        node_occasions.setup(len(self.G.nodes), self.preservation_duration)

        # Loading models and data 
        self.model = Net(self.dataset, self.data, device=self.device).to(self.device)
        self.scale_factors = scale_factors

        # Check initial model accuracy
        accs, _, _, _, _ = test(self.model, self.data)
        self.logger.load_model_log(accs)

        # Initialise adj matrix utils
        support1 = self.model.adj1
        support2 = self.model.adj2
        self.partial_adj_mask = support1
        adj_variables = [support1, support2]
        rho = 1e-3
        self.Z1 = self.U1 = self.Z2 = self.U2 = self.partial_adj_mask.new_zeros(self.partial_adj_mask.size()).to(self.device)
        self.model.adj1.requires_grad = True
        self.model.adj2.requires_grad = True
        self.adj_map = {"support1": support1, "support2": support2}
            
        # Define new loss function
        self.loss = lambda m, d: F.nll_loss(m()[d.train_mask], d.y[d.train_mask])
        self.admm_loss = lambda m, d: self.loss(m, d) + \
                    rho * (F.mse_loss(support1 + self.U1, self.Z1 + torch.eye(support1.shape[0]).to(self.device)) +
                    F.mse_loss(support2 + self.U2, self.Z2 + torch.eye(support2.shape[0]).to(self.device)))

        # Deine optimisers
        self.adj_optimizer = torch.optim.Adam(adj_variables,lr=self.adj_lr)
        self.weight_optimizer = torch.optim.Adam([
            dict(params=self.model.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.model.conv2.parameters(), weight_decay=0)
        ], lr=self.w_lr)

    def train(self):
        init_time = time.time()
        # Training 
        for j in range(self.times):
            for epoch in range(self.epochs):
                # for batch in gcnloader:
                t = time.time()
                self.model.train()
                self.adj_optimizer.zero_grad()
                self.weight_optimizer.zero_grad()
                # Calculate gradient			
                self.admm_loss(self.model, self.data).backward(retain_graph=True)
                
                # Update to correct gradient
                update_gradients_adj(self.adj_map, self.partial_adj_mask)

                # Use the optimizer to update adjacency matrix
                self.adj_optimizer.step()
                self.weight_optimizer.step() 

                time_per_epoch = time.time() - t
                    
                if epoch == 0 and j == 0:
                    train_memory = torch.cuda.max_memory_allocated(self.device)*2**(-20)

                accs, senss, specs, f1s, aucs = test(self.model, self.data)
                self.logger.train_time_log(j, epoch, time_per_epoch, accs, senss, specs, f1s, aucs)

                # Prune the graph in the first 20 epochs
                # Then only prune after every set epoch intervals
                if (epoch == (self.epochs - 1)):
                    print("Pruning the graph, epoch: ", (j + 1) * self.epochs, "acc: ", accs['test_mask'])
                    # Use learnt U1, Z1 and so on to prune
                    adj1 = self.model.adj1 
                    adj2 = self.model.adj2
                    adj1_shape = torch.eye(adj1.shape[0]).to(self.device)
                    adj2_shape = torch.eye(adj2.shape[0]).to(self.device)

                    non_zero_idx = np.count_nonzero(adj1.cpu().detach().numpy())

                    self.Z1 = adj1 - adj1_shape + self.U1
                    self.Z1 = dacp_prune_adj(self.Z1, non_zero_idx, self.prune_ratio, self.scale_factors, self.centrality, self.preserve_rate, self.device) - adj1_shape
                    self.U1 = self.U1 + (adj1 - adj1_shape - self.Z1)

                    self.Z2 = adj2 - adj2_shape + self.U2
                    self.Z2 = dacp_prune_adj(self.Z2, non_zero_idx, self.prune_ratio, self.scale_factors, self.centrality, self.preserve_rate, self.device) - adj2_shape
                    self.U2 = self.U2 + (adj2 - adj2_shape - self.Z2)

        # Logging final results
        accs, senss, specs, f1, aucs = test(self.model, self.data)
        self.logger.final_results_log(accs, senss, specs, f1, aucs)
        total_time = time.time() - init_time
        print(f"Training GPU Memory Usage: {train_memory} MB")
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(self.device)*2**(-20)} MB")
        print(f"total training time: {total_time}s")
        return accs["test_mask"]
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action="store_true", help="Use this flag to enable GPU usage. Default is False.")
    parser.add_argument('--total_epochs', type=int, default=400)
    parser.add_argument('--dest', type=str, default='.')
    parser.add_argument("--dataset_name", type=str, default="Cora")
    parser.add_argument('--w_lr', type=float, default=0.03)
    parser.add_argument('--adj_lr', type=float, default=0.001)
    parser.add_argument('--prune_ratio', type=int, default=50)
    parser.add_argument('--preserve_rate', type=int, default=90)
    parser.add_argument('--centrality', type=str, default='BC_DC')  
    parser.add_argument('--outer_k', type=int, default=0)
    parser.add_argument('--inner_k', type=int, default=0)
    parser.add_argument('--preserve_duration', type=int, default=3)
    parser.add_argument('--test_run', action="store_true", help="Use this flag to run the model with test masks. Default is False.")
    parser.add_argument('--ac_select', type=str, default='minmax')
    parser.add_argument('--BC_sf', type=float, default=1.0)
    parser.add_argument('--CC_sf', type=float, default=1.0)
    parser.add_argument('--DC_sf', type=float, default=1.0)
    parser.add_argument('--EC_sf', type=float, default=1.0)
    parser.add_argument('--direct_comparison', action="store_true", help="Use this flag to run the model using direct comparison")
    args = parser.parse_args()

    model = DACPSGCN(use_gpu=args.use_gpu, total_epochs=args.total_epochs, 
                dest=args.dest, dataset_name=args.dataset_name, w_lr=args.w_lr, 
                adj_lr=args.adj_lr, prune_ratio=args.prune_ratio, 
                preserve_rate=args.preserve_rate, centrality=args.centrality, 
                ac_select=args.ac_select)
    
    if args.direct_comparison:
        model.ssp_mask() 
    else:
        all_masks = dataset_manager.load_masks(args.dataset_name)
        outer_fold = f"outer_fold_{args.outer_k}"
        inner_fold = f"select_mask_{args.inner_k}"

        if args.test_run:
            test_mask = all_masks[outer_fold]["test_mask"]
        else:
            test_mask = all_masks[outer_fold][inner_fold]["val_mask"]

        model.load_masks(all_masks[outer_fold][inner_fold]["train_mask"],
                            test_mask)

    scale_factors = [args.BC_sf, args.CC_sf, args.DC_sf, args.EC_sf]
    model.set_up(scale_factors)
    model.train()
    model.save_experiment_results()


