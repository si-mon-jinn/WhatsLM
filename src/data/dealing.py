# Deals training data to training phase

import torch

class DataDealer():
    def __init__(self):

        return
    
    def get_batch(self):
        return
    
    def _split_data(self):
        return

class txtDealer(DataDealer):
    def __init__(self, encoded_data, batch_size: int = 1, block_size: int = 8): # Better to not have default batch and block sizes, they're not needed
        super().__init__()

        self.batch_size = batch_size
        self.block_size = block_size

        self.train_data = None
        self.val_data = None
        self._split_data(encoded_data)
        
        return
    
    def get_batch(self, batch_size: int = None, block_size: int = None, split: str = 'train', device: str = 'cpu'):
        if batch_size is None: batch_size = self.batch_size
        if block_size is None: block_size = self.block_size

        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data)-block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        x, y = x.to(device), y.to(device)
        
        return x, y
    
    def _split_data(self, data):
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        return
    
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
    
    def set_block_size(self, block_size: int):
        self.block_size = block_size
    
    def get_block_size(self) -> int:
        return self.block_size
    
    def get_batch_size(self) -> int:
        return self.batch_size