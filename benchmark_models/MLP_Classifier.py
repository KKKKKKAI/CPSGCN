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

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm = False, dropout = 0.0, 
                 drop_input = False):
        
        super().__init__()
        torch.manual_seed(1)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        
        self.linear_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        
        # Adding input layer
        self.linear_layers.append(torch.nn.Linear(in_channels, hidden_channels))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding hidden layers
        for i in range(num_layers-2):
            self.linear_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Adding output layer
        self.linear_layers.append(torch.nn.Linear(hidden_channels, out_channels))
    
    def forward(self, data):
        x = data.x
        # if using input dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        for i in range(self.num_layers-1): # exclude output layer
            x = self.linear_layers[i](x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.linear_layers[-1](x) # output layer
        
        return x


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

model = MLP(hidden_channels=args.hidden_units, num_layers=3, in_channels=dataset.num_features, out_channels=dataset.num_classes)

if args.use_gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

logger = Logger(args.total_epochs, args.dataset, args.dest)

net = NetWrapper(model, loss_function=nn.CrossEntropyLoss(), device=device, logger=logger)

net.train(data, max_epochs=args.total_epochs, optimizer=optimizer)
