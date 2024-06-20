import torch

from torch_geometric.nn import GCNConv 
from utils import *
from net_utils import *

import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.utils.checkpoint as checkpoint

class Net(torch.nn.Module): 
    def __init__(self, dataset, data, adj=(), device='cuda:0'):
        super(Net, self).__init__()
        self.data = data
        self.conv1 = GCNConv(dataset.num_features, 16, normalize=True)
        self.conv2 = GCNConv(16, dataset.num_classes, normalize=True)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.device = device
        
        if len(adj) == 0:
            self.adj1 = edge_to_adj(data.edge_index,data.edge_attr,data.num_nodes).to(device)
        else:
            self.adj1 = torch.from_numpy(adj).to(device)
        
        self.adj2 = self.adj1.clone().to(device)
    
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
        self.ei1, self.ew1 = adj_to_edge(self.adj1, self.device)
        self.ei2, self.ew2 = adj_to_edge(self.adj2, self.device)
        
        x = checkpoint.checkpoint(self.checkpoint1(self.conv1),x, use_reentrant=False)
        x = checkpoint.checkpoint(self.checkpoint3(),x, use_reentrant=False)
        x = F.dropout(x, training=self.training)
        
        x = checkpoint.checkpoint(self.checkpoint2(self.conv2),x, use_reentrant=False)
        return F.log_softmax(x, dim=1)

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)
    