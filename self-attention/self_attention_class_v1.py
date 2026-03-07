import torch
import torch.nn as nn

class SelfAttentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Wq = nn.Parameter(torch.rand(d_in, d_out))
        self.Wv = nn.Parameter(torch.rand(d_in, d_out))
        self.Wk = nn.Parameter(torch.rand(d_in, d_out))
    
    def froward(self, x):
        keys = x @ self.Wk 
        values = x @ self.Wv 
        queries = x @ self.Wq 
        
        attn_scores = queries @ keys 
        attn_weights = torch.softmax(attn_scores / keys.shape[1]**0.5, dim=-1) 
        
        context_vec = attn_weights @ values 
        return context_vec