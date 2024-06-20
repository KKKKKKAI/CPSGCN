from __future__ import division

import time
import os
import shutil
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import utils as ut
import psgd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import warnings

# Suppress the specific warning message
warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes")

import sys
import os

# Get the absolute path to the directory two levels up
parent_directory = os.path.join(os.path.dirname(__file__), '../../')
# Add the parent directory to the system path
sys.path.append(parent_directory)
from logger import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run(
    data, 
    dataset_name,
    model, 
    str_optimizer, 
    str_preconditioner, 
    epochs, 
    lr, 
    weight_decay, 
    early_stopping,  
    logger, 
    momentum,
    eps,
    update_freq,
    gamma,
    alpha,
    dest
    ):

    logger = Logger(epochs, dataset_name, dest)

    val_losses, accs, durations = [], [], []
    torch.manual_seed(42)

    data = data.to(device)

    model.to(device).reset_parameters()
    if str_preconditioner == 'KFAC':

        preconditioner = psgd.KFAC(
            model, 
            eps, 
            sua=False, 
            pi=False, 
            update_freq=update_freq,
            alpha=alpha if alpha is not None else 1.,
            constraint_norm=False
        )
    else: 
        preconditioner = None

    if str_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif str_optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    val_loss_history = []
    
    init_time = time.time()

    for epoch in range(1, epochs + 1):
        lam = (float(epoch)/float(epochs))**gamma if gamma is not None else 0.
        t = time.time()
        train(model, optimizer, data, preconditioner, lam)
        time_per_epoch = time.time() - t

        accs, senss, specs, f1s, losses, aucs = evaluate(model, data)
        eval_info = {}
        eval_info['epoch'] = int(epoch)
        eval_info['time'] = time.perf_counter() - t_start
        eval_info['eps'] = eps
        eval_info['update-freq'] = update_freq
        
        if epoch == 1:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)


        logger.train_time_log(epoch, time_per_epoch, accs, senss, specs, f1s, aucs)

        if gamma is not None:
            eval_info['gamma'] = gamma
        
        if alpha is not None:
            eval_info['alpha'] = alpha

        # if logger is not None:
        #     for k, v in eval_info.items():
        #         logger.add_scalar(k, v, global_step=epoch)

        val_loss_history.append(losses["test_mask"])
        if early_stopping > 0 and epoch > epochs // 2:
            tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
            if losses["test_mask"] > tmp.mean().item():
                break

    logger.final_results_log(accs, senss, specs, f1s, aucs)
    logger.save_experiment_results()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    durations.append(t_end - t_start)

    total_time = time.time() - init_time
    print(f"Training GPU Memory Usage: {train_memory} MB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
    print(f"total training time: {total_time}s")



def train(model, optimizer, data, preconditioner=None, lam=0.):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False
    
    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])

    loss.backward(retain_graph=True)
    if preconditioner:
        preconditioner.step(lam=lam)
    optimizer.step()

def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    accs, senss, specs, f1s, losses, aucs = dict(), dict(), dict(), dict(), dict(), dict()
    
    for key in ['train', 'test']:
        mask_name = '{}_mask'.format(key)
        mask = data[mask_name]
        losses[mask_name] = F.nll_loss(logits[mask], data.y[mask]).item()
        
        pred_probs = torch.softmax(logits[mask], dim=1)
    
		# Get the predicted class labels
        y_pred = pred_probs.max(1)[1].cpu()
        y_true = data.y[mask].cpu()
		
        accs[mask_name] = accuracy_score(y_true, y_pred)

        senss[mask_name] = recall_score(y_true, y_pred, average='macro')

        specs[mask_name] = precision_score(y_true, y_pred, average='macro', zero_division=0)

        f1s[mask_name] = f1_score(y_true, y_pred, average='macro')
        
        if len(torch.unique(y_true)) > 2:  # Multiclass case
            aucs[mask_name] = roc_auc_score(y_true, pred_probs.cpu(), multi_class='ovr')
        else:  # Binary case
            # For binary case, use probabilities of the positive class
            aucs[mask_name] = roc_auc_score(y_true, pred_probs[:, 1].cpu())


    return accs, senss, specs, f1s, losses, aucs

