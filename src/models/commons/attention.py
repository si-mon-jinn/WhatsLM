import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):
    def __init__(self, head_size: int, block_size: int, dropout: float):
        super().__init__()

        self.value_matrix = nn.Linear(head_size, head_size, bias=False)
        self.query_matrix = nn.Linear(head_size, head_size, bias=False)
        self.key_matrix = nn.Linear(head_size, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, toks): # toks: (B, T, C)
        B, T, C = toks.shape
        weights = (self.query_matrix(toks) @ self.key_matrix(toks).transpose(-2,-1))*C**-0.5 #USE THE RIGHT C!

        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out_toks = weights @ self.value_matrix(toks)

        return out_toks
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads:int, head_size: int, block_size: int, dropout:float):
        super().__init__()

        self.head_size = head_size


        self.heads = nn.ModuleList([AttentionHead(head_size=self.head_size, block_size=block_size, dropout=dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads*head_size, num_heads*head_size)
    
    def forward(self, toks): # toks: (B, T, C)
        # Not sure this is the best way of doing it
        out_toks = torch.concat([head(toks[:,:,n*self.head_size:(n+1)*self.head_size]) for n, head in enumerate(self.heads)], dim=-1)
        out_toks = self.linear(out_toks)

        return out_toks