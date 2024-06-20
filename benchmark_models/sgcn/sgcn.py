import time
import torch
import argparse

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

import numpy as np
import os.path as osp
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.utils.checkpoint as checkpoint
import torch_geometric.utils.num_nodes as geo_num_nodes


import sys
import os
parent_directory = os.path.join(os.path.dirname(__file__), '../../')
# Add the parent directory to the system path
sys.path.append(parent_directory)
from logger import SGCNLogger
from net_utils import test
import dataset_manager

def update_gradients_adj(grads_vars ,adj_p_mask):
    temp_grad_adj1 = 0
    var1 = None
    var2 = None
    temp_grad_adj2 = 0
    for key,var in grads_vars.items():
        grad = var.grad
        if key == "support1":
            adj_mask = adj_p_mask
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
            var1 = var
        if key == "support2":
            adj_mask = adj_p_mask
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj2 = temp_grad_adj + transposed_temp_grad_adj
            var2 = var
    grad_adj = (temp_grad_adj1 + temp_grad_adj2) / 4 # Why are we doing this?
    var1.grad = grad_adj
    var2.grad = grad_adj
    return [var1,var2]

def prune_adj(oriadj, non_zero_idx:int, percent:int) -> torch.Tensor:
	original_prune_num = int(((non_zero_idx - oriadj.shape[0]) / 2) * (percent / 100))
	adj = oriadj.cpu().detach().numpy()
	# print(f"Pruning {percent}%")
	low_adj = np.tril(adj, -1)
	non_zero_low_adj = low_adj[low_adj != 0]

	low_pcen = np.percentile(abs(non_zero_low_adj), percent)
	under_threshold = abs(low_adj) < low_pcen
	before = len(non_zero_low_adj)
	low_adj[under_threshold] = 0
	non_zero_low_adj = low_adj[low_adj != 0]
	after = len(non_zero_low_adj)

	rest_pruned = original_prune_num - (before - after)
	# print(adj.shape[0],original_prune_num,before,after, before-after)
	if rest_pruned > 0:
		mask_low_adj = (low_adj != 0)
		low_adj[low_adj == 0] = 2000000
		flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
		row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
		low_adj = np.multiply(low_adj, mask_low_adj)
		low_adj[row_indices, col_indices] = 0
	adj = low_adj + np.transpose(low_adj)
	adj = np.add(adj, np.identity(adj.shape[0]))
	return torch.from_numpy(adj)

def edge_to_adj(edge_index, edge_attr=None,num_nodes=None):
    row, col = edge_index

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1)
        assert edge_attr.size(0) == row.size(0)
    n_nodes = geo_num_nodes.maybe_num_nodes(edge_index, num_nodes)
    diff_adj = torch.zeros([n_nodes,n_nodes])
    diff_adj += torch.eye(diff_adj.shape[0])
    diff_adj[row,col] = edge_attr
    return diff_adj

def adj_to_edge(adj:torch.Tensor):
    new_adj = adj - torch.eye(adj.shape[0]).to(device)
    edge_index = (new_adj > 0).nonzero(as_tuple=False).t()
    row,col = edge_index
    edge_weight = new_adj[row,col].float()
    return (edge_index.to(device),edge_weight.to(device))

class Net(torch.nn.Module): 
    def __init__(self, dataset, data, args, adj=()):
        super(Net, self).__init__()
        self.data = data
        self.conv1 = GCNConv(dataset.num_features, 16,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes,
                             normalize=not args.use_gdc)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        # print(adj)
        if len(adj) == 0:
            self.adj1 = edge_to_adj(data.edge_index,data.edge_attr,data.num_nodes).to(device)
        else:
            self.adj1 = torch.from_numpy(adj).to(device)
        # self.adj1 = torch.eye(data.num_nodes)
        self.adj2 = self.adj1.clone().to(device)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
    
    def checkpoint1(self, module):
        def forward1(*inputs):
            return module(inputs[0], self.ei1, self.ew1)
        return forward1
    
    def checkpoint3(self):
        def forward3(*inputs):
            return F.relu(inputs[0])
        return forward3
    def checkpoint2(self, module):
        def forward2(*inputs):
            return module(inputs[0], self.ei2, self.ew2)
        return forward2

    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr
        self.ei1, self.ew1 = adj_to_edge(self.adj1)
        self.ei2, self.ew2 = adj_to_edge(self.adj2)
        x = checkpoint.checkpoint(self.checkpoint1(self.conv1),x, self.dummy)
        x = checkpoint.checkpoint(self.checkpoint3(),x,self.dummy)
        x = F.dropout(x, training=self.training)
        x = checkpoint.checkpoint(self.checkpoint2(self.conv2),x, self.dummy)
        return F.log_softmax(x, dim=1)

def train(model,data):
    model.train()
    optimizer.zero_grad()

    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

############# Setting Up device controls ############   
parser = argparse.ArgumentParser()
parser.add_argument('--ratio_weight', type=int, default=10)
parser.add_argument('--ratio_graph', type=int, default=10)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--total_epochs', type=int, default=400)
parser.add_argument("--dataset", type=str, default="Cora")
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--dest', type=str, default='.')
parser.add_argument('--w_lr', type=float, default=0.03)
parser.add_argument('--adj_lr', type=float, default=0.001)
parser.add_argument('--outer_k', type=int, default=0)
parser.add_argument('--inner_k', type=int, default=0)
parser.add_argument('--test_run', action="store_true", help="Use this flag to run the model with test masks. Default is False.")
parser.add_argument('--use_gpu', action="store_true", help="Use this flag to enable GPU usage. Default is False.")
parser.add_argument('--direct_comparison', action="store_true", help="Use this flag to run the model using direct comparison")




args = parser.parse_args()

if(args.use_gpu == False):
    device = torch.device('cpu')
    print("using cpu... ")
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using gpu... ")

times = int(args.total_epochs / args.epochs)

logger = SGCNLogger(times, args.epochs, args.dataset, args.ratio_graph, args.dest)

dataset = Planetoid(root='data/Planetoid/'+args.dataset, name=args.dataset, transform=T.NormalizeFeatures())

all_masks = dataset_manager.load_masks(args.dataset)
outer_fold = f"outer_fold_{args.outer_k}"
inner_fold = f"select_mask_{args.inner_k}"

if args.test_run:
    dataset[0].test_mask = all_masks[outer_fold]["test_mask"]
else:
    dataset[0].test_mask = all_masks[outer_fold][inner_fold]["val_mask"]

dataset[0].train_mask = all_masks[outer_fold][inner_fold]["train_mask"]

data = dataset[0]
data.to(device)
# Loading models and data 
model = Net(dataset, data, args).to(device)

# Check initial model accuracy
accs, _, _, _, _ = test(model, data)
logger.load_model_log(accs)

# Initialise adj matrix utils
support1 = model.adj1
support2 = model.adj2
partial_adj_mask = support1
adj_variables = [support1, support2]
rho = 1e-3
Z1 = U1 = Z2 = U2 = partial_adj_mask.new_zeros(partial_adj_mask.size()).to(device)
model.adj1.requires_grad = True
model.adj2.requires_grad = True
adj_map = {"support1": support1, "support2": support2}
	
# Define new loss function
loss = lambda m, d: F.nll_loss(m()[d.train_mask], d.y[d.train_mask])
admm_loss = lambda m, d: loss(m, d) + \
            rho * (F.mse_loss(support1 + U1, Z1 + torch.eye(support1.shape[0]).to(device)) +
            F.mse_loss(support2 + U2, Z2 + torch.eye(support2.shape[0]).to(device)))

# Deine optimisers
adj_optimizer = torch.optim.Adam(adj_variables,lr=args.adj_lr)
weight_optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.w_lr)

init_time = time.time()

# Training 
for j in range(times):
    for epoch in range(args.epochs):
        # for batch in gcnloader:
        t = time.time()
        model.train()
        adj_optimizer.zero_grad()
        weight_optimizer.zero_grad()
        # Calculate gradient			
        admm_loss(model, data).backward(retain_graph=True)

        # Update to correct gradient
        update_gradients_adj(adj_map, partial_adj_mask)

        # Use the optimizer to update adjacency matrix
        adj_optimizer.step()
        weight_optimizer.step() 

        time_per_epoch = time.time() - t
        if epoch == 0 and j == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)

        # Prune the graph in the first 20 epochs
        # Then only prune after every set epoch intervals
        if (epoch == (args.epochs - 1)):
            print("Pruning the graph, epoch: ", (j + 1) * times, "acc: ", accs['test_mask'])
            # Use learnt U1, Z1 and so on to prune
            adj1,adj2 = model.adj1, model.adj2
            adj1_shape, adj2_shape = torch.eye(adj1.shape[0]).to(device), torch.eye(adj2.shape[0]).to(device)

            non_zero_idx = np.count_nonzero(adj1.cpu().detach().numpy())

            Z1 = adj1 - adj1_shape + U1
            Z1 = prune_adj(Z1, non_zero_idx, args.ratio_graph).to(device) - adj1_shape
            U1 = U1 + (adj1 - adj1_shape - Z1)

            Z2 = adj2 - adj2_shape + U2
            Z2 = prune_adj(Z2, non_zero_idx, args.ratio_graph).to(device) - adj2_shape
            U2 = U2 + (adj2 - adj2_shape - Z2)

        accs, senss, specs, f1s, aucs = test(model, data)
        logger.train_time_log(j, epoch, time_per_epoch, accs, senss, specs, f1s, aucs)

total_time = time.time() - init_time
print(f"Training GPU Memory Usage: {train_memory} MB")
print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
print(f"total training time: {total_time}s")


# global weight pruning
total = 0
for m in model.modules():
	if isinstance(m, GCNConv):
		total += m.weight.data.numel()
conv_weights = torch.zeros(total)
index = 0
for m in model.modules():
	if isinstance(m, GCNConv):
		size = m.weight.data.numel()
		conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
		index += size
y, i = torch.sort(conv_weights)
thre_index = int(total * args.ratio_weight / 100)
thre = y[thre_index]
pruned = 0
zero_flag = False
for k, m in enumerate(model.modules()):
	if isinstance(m, GCNConv):
		weight_copy = m.weight.data.abs().clone()
		mask = weight_copy.gt(thre).float().to(device)
		pruned = pruned + mask.numel() - torch.sum(mask)
		m.weight.data.mul_(mask)
		if int(torch.sum(mask)) == 0:
				zero_flag = True

# Logging final results
accs, senss, specs, f1, aucs = test(model, data)
logger.final_results_log(accs, senss, specs, f1, aucs)

logger.save_experiment_results()
