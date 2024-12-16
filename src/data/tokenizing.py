import torch

class Tokenizer():
    def __init__(self, clean_data: str):    
        self.vocab = self.comp_vocab(clean_data)
        self.vocab_size = len(self.vocab)

        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}

        return
    
    def encode(self, s):
        return torch.tensor([self.stoi[ch] for ch in s])
    
    def decode(self, i):
        return ''.join([self.itos[ch.item()] for ch in i])
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def comp_vocab(self, clean_data: str):
        return sorted(list(set(clean_data)))

