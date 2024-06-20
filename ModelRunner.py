from device_control import device_control
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T

class ModelRunner():
    def __init__(self, 
                 use_gpu=True,
                 epochs=400,
                 dest=None,
                 dataset_name="Cora",
                 w_lr=0.01,
                 root='data/Planetoid/',):
        
        # set up device controls    
        self.device_control = device_control()
        self.device_control.set_gpu_status(use_gpu)
        self.device = self.device_control.get_device()

        self.epochs = epochs
        self.dest = dest

        self.dataset = Planetoid(root=root+dataset_name, name=dataset_name, transform=T.NormalizeFeatures())
        self.data = self.dataset[0]

        self.w_lr = w_lr
    
    def getLogger(self):
        return self.logger


