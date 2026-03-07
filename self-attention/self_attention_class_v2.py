import torch
import torch.nn as nn

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def froward(self, x):
        keys = x @ self.Wk 
        values = x @ self.Wv 
        queries = x @ self.Wq 
        
        attn_scores = queries @ keys 
        attn_weights = torch.softmax(attn_scores / keys.shape[1]**0.5, dim=-1) 
        
        context_vec = attn_weights @ values 
        return context_vec