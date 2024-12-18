# Everything needed to run a forward pass for the architecture

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..commons.attention import MultiHeadAttention

class MHA(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_heads: int, head_size: int, block_size: int):
        super().__init__()

        self.block_size = block_size

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.pos_embedd = nn.Embedding(self.block_size, embed_size)

        self.out_layer = nn.Linear(embed_size, vocab_size)

        self.multi_head = MultiHeadAttention(num_heads=num_heads, head_size=head_size, block_size=self.block_size)
    
    def forward(self, toks, next_toks = None): # toks: (B, T), next_tok (B, T)
        T = toks.shape[1]
        out_toks = self.embeddings(toks[:,-T:]) + self.pos_embedd(torch.arange(0,T, dtype=torch.int, device='cuda'))# (B, T, C)

        out_toks = out_toks + self.multi_head(out_toks)

        out_toks = self.out_layer(out_toks)
        
        if next_toks == None:
            loss = None
        else:
            B, T, C = out_toks.shape
            loss = F.cross_entropy(out_toks.reshape(B*T,C), next_toks.view(B*T))

        return out_toks, loss
    
    def generate(self, idx, max_new_tokens):
        self.eval()
        
        for _ in range(max_new_tokens):
            cropped_idx = idx[:, -self.block_size:]
            
            logits, _ = self(cropped_idx)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx