import torch 

class device_control():    
    def __init__(self):
        self.use_gpu = True

    def set_gpu_status(self, status):
        self.use_gpu = status
        if (self.use_gpu):
            print("using gpu... ")
        else:
            print("using cpu... ")

        
    def use_gpu_status(self):
        return self.use_gpu

    def get_device(self):
        if(self.use_gpu_status() == False):
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device

