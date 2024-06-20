#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.transforms as T
from modules import *

from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from torch_geometric.datasets import Planetoid
from NetWrapper import NetWrapper
from torch import nn
import sys
import os

# Get the absolute path to the directory two levels up
parent_directory = os.path.join(os.path.dirname(__file__), '../')

# Add the parent directory to the system path
sys.path.append(parent_directory)

import dataset_manager
from logger import Logger


class DeepMultisets(torch.nn.Module):

    def __init__(self, dim_features, dim_target, hidden_units):
        super(DeepMultisets, self).__init__()

        self.fc_vertex = Linear(dim_features, hidden_units)
        self.fc_global1 = Linear(hidden_units, hidden_units)
        self.fc_global2 = Linear(hidden_units, dim_target)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.fc_vertex(x))
        x = self.aggregate(x, edge_index)
        x = F.relu(self.fc_global1(x))
        x = self.fc_global2(x)
        return x

    
    def aggregate(self, x, edge_index):
        # Perform mean pooling to aggregate information from neighboring nodes
        row, col = edge_index
        num_nodes = x.size(0)
        
        # Compute mean of neighboring node features
        row_sum = torch.zeros((num_nodes, x.size(1)), device=x.device)
        row_count = torch.zeros(num_nodes, device=x.device)
        
        row_sum.index_add_(0, row, x[col])
        row_count.index_add_(0, row, torch.ones_like(col, dtype=torch.float))
        
        row_count[row_count == 0] = 1  # Avoid division by zero
        aggregated = row_sum / row_count.view(-1, 1)
        
        return aggregated
        
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_units', type=int, default=32)
parser.add_argument('--total_epochs', type=int, default=400)
parser.add_argument("--dataset", type=str, default="Cora")
parser.add_argument('--dest', type=str, default='.')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--outer_k', type=int, default=0)
parser.add_argument('--inner_k', type=int, default=0)
parser.add_argument('--test_run', action="store_true", help="Use this flag to run the model with test masks. Default is False.")
parser.add_argument('--use_gpu', action="store_true", help="Use this flag to enable GPU usage. Default is False.")
args = parser.parse_args()


dataset = Planetoid(root='data/Planetoid/'+args.dataset, name=args.dataset, transform=T.NormalizeFeatures())

all_masks = dataset_manager.load_masks(args.dataset)
outer_fold = f"outer_fold_{args.outer_k}"
inner_fold = f"select_mask_{args.inner_k}"

data = dataset[0]

if args.test_run:
    data.test_mask = all_masks[outer_fold]["test_mask"]
else:
    data.test_mask = all_masks[outer_fold][inner_fold]["val_mask"]

data.train_mask = all_masks[outer_fold][inner_fold]["train_mask"]

model = DeepMultisets(dataset.num_features, dataset.num_classes, args.hidden_units)

if args.use_gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

logger = Logger(args.total_epochs, args.dataset, args.dest)

net = NetWrapper(model, loss_function=F.nll_loss, device=device, logger=logger)

net.train(data, max_epochs=args.total_epochs, optimizer=optimizer)
