# From loaded training data to interrupt conditions (evaluation, fixed epoch number)
# Load classes based on a config file

import torch
import json

from ..models.model_base import ModelLoader
from ..data.dealing import DataDealer

class ModelTrainer():
    def __init__(self, model_loader: ModelLoader, dealer: DataDealer, json_path: str, device: str = 'cpu'):
        self.device = device
        self.model_loader = model_loader

        self.model = self.model_loader.model
        self.module = self.model.model
        self.block_size = self.model.block_size

        self.get_batch = dealer.get_batch

        self.params = self._load_parameters(json_path)

        self.batch_size = self.params['batch_size']
        self.snap_freq = self.params['snapshots_freq']
        self.log_freq = self.params['log_freq']

        self.learn_rate = self.params['learning_rate']

        self.optimizer = torch.optim.AdamW(self.module.parameters(), 
                                           lr=self.learn_rate)

        return
    
    def train(self, steps: int, batch_size: int = 0):
        if batch_size == 0: batch_size = self.batch_size
        for step in range(steps):
            xb, yb = self.get_batch(split='train', 
                                    batch_size=batch_size, 
                                    block_size=self.block_size,
                                    device=self.device)

            _, loss = self.module(xb, yb)

            self.optimizer.zero_grad(set_to_none=True) # put gradients to zero
            loss.backward() # get gradient
            self.optimizer.step() # update parameters

            self.model.curr_steps+=1

            if (step+1) % self.log_freq == 0:
                mloss = self.get_loss(self.module, split = 'train', batch_size=10).item()
                mvloss = self.get_loss(self.module, split = 'valid', batch_size=10).item()

                print(f'{step+1}: {mloss:.4f} {mvloss:.4f}')
            
            if (step+1) % self.snap_freq == 0:
                self.model.model_loss += [self.get_loss(self.module, split = 'train', batch_size=1000).item()] #model_loss
                self.model.model_vloss += [self.get_loss(self.module, split = 'valid', batch_size=1000).item()] #model_vloss
                self._save_snapshot()

        if steps % self.snap_freq != 0:
            self.model.model_loss += [self.get_loss(self.module, split = 'train', batch_size=1000).item()] #model_loss
            self.model.model_vloss += [self.get_loss(self.module, split = 'valid', batch_size=1000).item()] #model_vloss
            self._save_snapshot()


    def get_loss(self, model, split = 'train', batch_size = 32):
        model.eval()
        
        xb, yb = self.get_batch(split=split, batch_size=batch_size, block_size=self.block_size, device=self.device)

        _, loss = model(xb, yb)

        model.train()
        return loss
    
    def _load_parameters(self, json_path: str) -> dict[str, any]:
        with open(json_path, 'r') as json_file:
            return json.loads(json_file.read())['training_handler']
        
    def _save_snapshot(self):
        #print('... saving snapshot ...')
        self.model_loader.save()
        return







'''from matplotlib import pyplot as plt


figure, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(d_mh_model_vloss, label='valid')
ax1.plot(d_mh_model_loss, label='train')
ax2.plot((torch.Tensor(d_mh_model_vloss)-torch.Tensor(d_mh_model_loss))*1)
figure.legend()
plt.plot()'''