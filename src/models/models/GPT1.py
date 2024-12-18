# Build a GPT like language model
# 
# Model architecture
#   - Feature and position embeddings (embed_size)
#   - Dropout
#   - Transformer block xN (num_blocks)
#       - LayerNorm
#       - Multi head attention (num_heads, head_size; check against embed_size?)
#           - Head xnum_heads
#               - Q x K
#               - Masking
#               - Softmax
#               - Dropout
#               - x V
#       - Linear
#       - Dropout
#       - LayerNorm( Previous + Transformer block input residual)
#       - Linear
#       - GeLU
#       - Linear
#       - Dropout + Residual from last LayerNorm
#   - LayerNorm
#   - Linear
#   - SoftMax

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..commons.attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, embed_size:int, gelu_size:int, dropout:float):
        super().__init__()

        self.layernorm = LayerNorm(embed_size=embed_size)

        self.linearpre = nn.Linear(embed_size, gelu_size)
        self.gelu = nn.GELU()
        self.linearpost = nn.Linear(gelu_size, embed_size)


        self.dropout = nn.Dropout(dropout)

    def forward(self, toks):
        toks = self.layernorm(toks)

        out_toks = self.linearpre(toks)
        out_toks = self.gelu(out_toks)
        out_toks = self.linearpost(out_toks)

        out_toks = toks + self.dropout(out_toks)

        return out_toks

class TransformerBlock(nn.Module):
    def __init__(self, embed_size:int, num_heads:int, head_size:int, block_size:int, dropout:float):
        super().__init__()

        self.layernorm_preheads = LayerNorm(embed_size=embed_size)
        
        #self.linear_preheads = nn.Linear()
        self.multi_head = MultiHeadAttention(num_heads=num_heads, head_size=head_size, block_size=block_size, dropout=dropout)
        self.linear_postheads = nn.Linear(num_heads*head_size, embed_size)
        self.dropout_postheads = nn.Dropout(dropout)

        self.ffwd = FeedForward(embed_size=embed_size, gelu_size=embed_size, dropout=dropout) # FIXME gelu_size should be set explicitely


    def forward(self, toks):

        out_toks = self.layernorm_preheads(toks)
        #out_toks = self.linear_preheads(out_toks)
        out_toks = self.multi_head(out_toks)
        out_toks = self.linear_postheads(out_toks)
        out_toks = toks + self.dropout_postheads(out_toks)

        out_toks = self.ffwd(out_toks)

        return out_toks

class LayerNorm(nn.Module):
    def __init__(self, embed_size:int, eps=1e-5):
        super().__init__()

        self.register_buffer('eps', torch.Tensor([eps]))

        self.gamma = nn.Parameter(torch.ones((embed_size,)))
        self.beta = nn.Parameter(torch.zeros((embed_size,)))
    
    def forward(self, toks): # B, T, C
        # Put on a gaussian centered in zero along the channels?
        mean = toks.mean(dim=-1, keepdim=True) # (B, T, 1)
        sigm = torch.sqrt(((toks-mean)**2).mean(dim=-1, keepdim=True)+self.eps) # (B, T, C) - (B, T, 1)

        out_toks = (toks-mean)/sigm*self.gamma+self.beta     

        return out_toks



class GPT1(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_heads: int, head_size: int, block_size: int, num_blocks: int, dropout: float=0.2):
        super().__init__()

        self.block_size = block_size

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.pos_embedd = nn.Embedding(self.block_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[TransformerBlock(embed_size=embed_size, num_heads=num_heads, head_size=head_size, block_size=self.block_size, dropout=dropout) for _ in range(num_blocks)])

        self.layernorm = LayerNorm(embed_size=embed_size)

        self.out_layer = nn.Linear(embed_size, vocab_size)

        
    
    def forward(self, toks, next_toks = None): # toks: (B, T), next_tok (B, T)
        T = toks.shape[1]
        out_toks = self.embeddings(toks[:,-T:]) + self.pos_embedd(torch.arange(0,T, dtype=torch.int, device='cuda'))# (B, T, C)

        out_toks = self.dropout(out_toks)

        out_toks = self.blocks(out_toks)

        out_toks = self.layernorm(out_toks)

        out_toks = self.out_layer(out_toks)


        
        if next_toks == None:
            loss = None
        else:
            B, T, C = out_toks.shape
            loss = F.cross_entropy(out_toks.reshape(B*T,C), next_toks.view(B*T)) # is sofmax already inside here?

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