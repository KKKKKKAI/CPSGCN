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
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_max_pool
import argparse
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from NetWrapper import NetWrapper
import sys
import os

# Get the absolute path to the directory two levels up
parent_directory = os.path.join(os.path.dirname(__file__), '../')

# Add the parent directory to the system path
sys.path.append(parent_directory)

import dataset_manager
from logger import Logger


class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, aggregation):
        super(GraphSAGENet, self).__init__()
        self.aggregation = aggregation

        if aggregation == 'max':
            self.fc_max = nn.Linear(hidden_channels, hidden_channels)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = in_channels if i == 0 else hidden_channels
            conv = SAGEConv(dim_input, hidden_channels, aggr=aggregation)
            self.layers.append(conv)

        self.fc1 = nn.Linear(num_layers * hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = F.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_units', type=int, default=32)
parser.add_argument('--total_epochs', type=int, default=400)
parser.add_argument("--dataset", type=str, default="Cora")
parser.add_argument('--dest', type=str, default='.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--aggregation', type=str, default="max")
parser.add_argument('--outer_k', type=int, default=0)
parser.add_argument('--inner_k', type=int, default=0)
parser.add_argument('--test_run', action="store_true", help="Use this flag to run the model with test masks. Default is False.")
parser.add_argument('--use_gpu', action="store_true", help="Use this flag to enable GPU usage. Default is False.")
parser.add_argument('--direct_comparison', action="store_true", help="Use this flag to run the model using direct comparison")
args = parser.parse_args()

dataset = Planetoid(root='data/Planetoid/'+args.dataset, name=args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.direct_comparison:
    train_mask = data.train_mask.fill_(False)
    train_mask[:data.num_nodes - 1000] = 1
    val_mask = data.val_mask.fill_(False)
    val_mask[data.num_nodes - 1000:data.num_nodes - 500] = 1
    test_mask = data.test_mask.fill_(False)
    test_mask[data.num_nodes - 500:] = 1
    data.train_mask = train_mask
    data.test_mask = test_mask
else:
    all_masks = dataset_manager.load_masks(args.dataset)
    outer_fold = f"outer_fold_{args.outer_k}"
    inner_fold = f"select_mask_{args.inner_k}"

    if args.test_run:
        data.test_mask = all_masks[outer_fold]["test_mask"]
    else:
        data.test_mask = all_masks[outer_fold][inner_fold]["val_mask"]

    data.train_mask = all_masks[outer_fold][inner_fold]["train_mask"]


configs = {
    "num_layers": args.num_layers,
    "hidden_units": args.hidden_units,
    "aggregation": 'max'
}

# model = GraphSAGENet(dataset.num_features, args.hidden_units, dataset.num_classes)
model = GraphSAGENet(dataset.num_features, args.hidden_units, dataset.num_classes, args.num_layers, 'max')
if args.use_gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=args.lr)

logger = Logger(args.total_epochs, args.dataset, args.dest)

net = NetWrapper(model, loss_function=nn.CrossEntropyLoss(), device=device, logger=logger)

net.train(data, max_epochs=args.total_epochs, optimizer=optimizer)
