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
import time
from datetime import timedelta
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import torch.nn.functional as F

import torch

class NetWrapper:
    def __init__(self, model, loss_function, device='cpu', logger=None):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.logger = logger
        self.model.to(self.device)

    def _train(self, data, optimizer, clipping=None):
        self.model.train()

        data = data.to(self.device)
        optimizer.zero_grad()
        output = self.model(data)
        
        label = output.max(1)[1]
        label[data.train_mask] = data.y[data.train_mask]
        label.requires_grad = False

        label = label.long()
        output = output.float()
        
        loss = F.nll_loss(output[data.train_mask], label[data.train_mask])
        loss.backward()

        if clipping is not None:  # Clip gradient before updating weights
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping)
        optimizer.step()

        return loss
        

    def classify_graphs(self, data):
        self.model.eval()

        data = data.to(self.device)
        output = self.model(data).float()
        truth = data.y.long()
        accs, senss, specs, f1s, losses, aucs = dict(), dict(), dict(), dict(), dict(), dict()
        
        for key in ['train', 'test']:
            mask_name = '{}_mask'.format(key)
            mask = data[mask_name]
            loss = F.nll_loss(output[mask], truth[mask])
            losses[mask_name] = loss.item()
            pred_probs = torch.softmax(output[mask], dim=1)

            pred = pred_probs.max(1)[1].cpu()

            accs[mask_name] = accuracy_score(truth[mask].cpu(), pred.cpu())

            senss[mask_name] = recall_score(truth[mask].cpu(), pred.cpu(), average='macro')

            specs[mask_name] = precision_score(truth[mask].cpu(), pred.cpu(), average='macro', zero_division=0)

            f1s[mask_name] = f1_score(truth[mask].cpu(), pred.cpu(), average='macro')

            if len(torch.unique(truth[mask].cpu())) > 2:  # Multiclass case
                aucs[mask_name] = roc_auc_score(truth[mask].cpu(), pred_probs.detach().cpu(), multi_class='ovr')
            else:  # Binary case
                # For binary case, use probabilities of the positive class
                aucs[mask_name] = roc_auc_score(truth[mask].cpu(), pred_probs[:, 1].cpu())

        return accs, senss, specs, f1s, losses, aucs

    def train(self, data, max_epochs=400, optimizer=torch.optim.Adam, scheduler=None, clipping=None):
        init_time = time.time()

        for epoch in range(0, max_epochs):

            start = time.time()
            self._train(data, optimizer, clipping)
            time_per_epoch = time.time() - start
            if epoch == 0:
                train_memory = torch.cuda.max_memory_allocated(self.device)*2**(-20)

            if scheduler is not None:
                scheduler.step(epoch)

            accs, senss, specs, f1s, _, aucs = self.classify_graphs(data)
            self.logger.train_time_log(epoch, time_per_epoch, accs, senss, specs, f1s, aucs)

        self.logger.final_results_log(accs, senss, specs, f1s, aucs)
        self.logger.save_experiment_results()

        total_time = time.time() - init_time
        print(f"Training GPU Memory Usage: {train_memory} MB")
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(self.device)*2**(-20)} MB")
        print(f"total training time: {total_time}s")

