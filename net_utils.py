import time
import torch

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from utils import *
from centrality import *


import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes

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

def adj_to_edge(adj:torch.Tensor, device):
    new_adj = adj - torch.eye(adj.shape[0]).to(device)
    edge_index = (new_adj > 0).nonzero(as_tuple=False).t()
    row,col = edge_index
    edge_weight = new_adj[row,col].float()
    return (edge_index.to(device),edge_weight.to(device))

def train(model, data, optimizer):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    time_per_epoch = time.time() - t
    print("Time per epoch: " + str(time_per_epoch))


@torch.no_grad()
def test(model, data):
	model.eval()
	accs, senss, specs, f1s, aucs = dict(), dict(), dict(), dict(), dict()
	logits = model()
	masks = ['train_mask', 'test_mask']
	for i,(_, data_mask) in enumerate(data('train_mask', 'test_mask')):
		pred_probs = torch.softmax(logits[data_mask], dim=1)
    
		# Get the predicted class labels
		y_pred = pred_probs.max(1)[1].cpu()

		# pred = logits[data_mask].max(1)[1]
		y_true = data.y[data_mask].cpu()
		# y_pred = pred.cpu()

		accs[masks[i]] = accuracy_score(y_true, y_pred)

		senss[masks[i]] = recall_score(y_true, y_pred, average='macro')

		specs[masks[i]] = precision_score(y_true, y_pred, average='macro', zero_division=0)

		f1s[masks[i]] = f1_score(y_true, y_pred, average='macro')

		if len(torch.unique(y_true)) > 2:  # Multiclass case
			aucs[masks[i]] = roc_auc_score(y_true, pred_probs.cpu(), multi_class='ovr')
		else:  # Binary case
			# For binary case, use probabilities of the positive class
			aucs[masks[i]] = roc_auc_score(y_true, pred_probs[:, 1].cpu())

	return accs, senss, specs, f1s, aucs

####################### Now, learn the graph with the new weights
def update_gradients_adj(grads_vars, partial_adj_mask):
	var1 = None
	var2 = None
	temp_grad_adj1 = 0
	temp_grad_adj2 = 0
	for key, var in grads_vars.items():
		grad = 0 if var.grad == None else var.grad
		if key == "support1":
			temp_grad_adj = partial_adj_mask * grad
			transposed_temp_grad_adj = torch.transpose(temp_grad_adj, 1, 0)
			temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
			var1 = var
		if key == "support2":
			temp_grad_adj = partial_adj_mask * grad
			transposed_temp_grad_adj = torch.transpose(temp_grad_adj, 1, 0)
			temp_grad_adj2 = temp_grad_adj + transposed_temp_grad_adj
			var2 = var
	grad_adj = (temp_grad_adj1 + temp_grad_adj2) / 4 
	var1.grad = grad_adj
	var2.grad = grad_adj
	return [var1, var2]

def prune_adj(oriadj:torch.Tensor, non_zero_idx, percent, centrality, preserve_rate, device) -> torch.Tensor:
	adj = oriadj.detach().cpu().numpy()
	
	########## Initial pruning base on percentile
	prune_num_required = int(((non_zero_idx - oriadj.shape[0]) / 2) * (percent / 100))
	low_adj = np.tril(adj, -1)
	low_adj_copy = low_adj.copy() 
	non_zero_low_adj = low_adj[low_adj != 0]
	min_pcentile = np.percentile(abs(non_zero_low_adj), percent)
	under_threshold = abs(low_adj) < min_pcentile # Matrix of ones to prune
	before = len(non_zero_low_adj)
	low_adj[under_threshold] = 0
	
	pruned_G = nx.from_numpy_matrix(low_adj_copy)
	
	########## Check pruning ratio after first pruning
	non_zero_low_adj = low_adj[low_adj != 0]
	after = len(non_zero_low_adj)
	remain_to_prune = prune_num_required - (before - after)

	if remain_to_prune > 0:
		mask_low_adj = (low_adj != 0)
		max_val = np.max(abs(low_adj))
		low_adj[low_adj == 0] = max_val
		flat_indices = np.argpartition(low_adj.ravel(), remain_to_prune - 1)[:remain_to_prune]
		row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
		low_adj = np.multiply(low_adj, mask_low_adj)
		low_adj[row_indices, col_indices] = 0
	
	adj = centrality_preservation(pruned_G, centrality, preserve_rate, low_adj, low_adj_copy)

	adj = adj + np.transpose(adj) + np.identity(adj.shape[0])
	return torch.from_numpy(adj).to(device)

def acp_prune_adj(oriadj:torch.Tensor, non_zero_idx, percent, centrality, preserve_rate, device, ac_selection="minmax") -> torch.Tensor:
	adj = oriadj.detach().cpu().numpy()
	
	########## Initial pruning base on percentile
	prune_num_required = int(((non_zero_idx - oriadj.shape[0]) / 2) * (percent / 100))
	low_adj = np.tril(adj, -1)
	low_adj_copy = low_adj.copy() 
	non_zero_low_adj = low_adj[low_adj != 0]
	min_pcentile = np.percentile(abs(non_zero_low_adj), percent)
	under_threshold = abs(low_adj) < min_pcentile # Matrix of ones to prune
	before = len(non_zero_low_adj)
	low_adj[under_threshold] = 0
	
	pruned_G = nx.from_numpy_matrix(low_adj_copy)
	
	########## Check pruning ratio after first pruning
	non_zero_low_adj = low_adj[low_adj != 0]
	after = len(non_zero_low_adj)
	remain_to_prune = prune_num_required - (before - after)

	if remain_to_prune > 0:
		mask_low_adj = (low_adj != 0)
		max_val = np.max(abs(low_adj))
		low_adj[low_adj == 0] = max_val
		flat_indices = np.argpartition(low_adj.ravel(), remain_to_prune - 1)[:remain_to_prune]
		row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
		low_adj = np.multiply(low_adj, mask_low_adj)
		low_adj[row_indices, col_indices] = 0
	
	adj = additive_centrality_preservation(pruned_G, centrality, preserve_rate, low_adj, low_adj_copy, ac_selection)

	adj = adj + np.transpose(adj) + np.identity(adj.shape[0])
	return torch.from_numpy(adj).to(device)



def dacp_prune_adj(oriadj:torch.Tensor, non_zero_idx, percent, scale_factors, centrality, preserve_rate, device) -> torch.Tensor:
	adj = oriadj.detach().cpu().numpy()
	
	########## Initial pruning base on percentile
	prune_num_required = int(((non_zero_idx - oriadj.shape[0]) / 2) * (percent / 100))
	low_adj = np.tril(adj, -1)
	low_adj_copy = low_adj.copy() 
	non_zero_low_adj = low_adj[low_adj != 0]
	min_pcentile = np.percentile(abs(non_zero_low_adj), percent)
	under_threshold = abs(low_adj) < min_pcentile # Matrix of ones to prune
	before = len(non_zero_low_adj)
	low_adj[under_threshold] = 0
	
	pruned_G = nx.from_numpy_matrix(low_adj_copy)
	
	########## Check pruning ratio after first pruning
	non_zero_low_adj = low_adj[low_adj != 0]
	after = len(non_zero_low_adj)
	remain_to_prune = prune_num_required - (before - after)

	if remain_to_prune > 0:
		mask_low_adj = (low_adj != 0)
		max_val = np.max(abs(low_adj))
		low_adj[low_adj == 0] = max_val
		flat_indices = np.argpartition(low_adj.ravel(), remain_to_prune - 1)[:remain_to_prune]
		row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
		low_adj = np.multiply(low_adj, mask_low_adj)
		low_adj[row_indices, col_indices] = 0
	
	adj = dynamic_additive_centrality_preservation(pruned_G, scale_factors, centrality, preserve_rate, low_adj, low_adj_copy, device)

	adj = adj + np.transpose(adj) + np.identity(adj.shape[0])
	return torch.from_numpy(adj).to(device)