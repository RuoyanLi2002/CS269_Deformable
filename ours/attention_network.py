import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

class SelfAttentionBlock(nn.Module):
    def __init__(self, latent_size, num_heads=4):
        super(SelfAttentionBlock, self).__init__()
        
        self.mha = nn.MultiheadAttention(embed_dim=latent_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_size)
       
        self.mlp = nn.Sequential(nn.Linear(latent_size, latent_size),)
        self.norm2 = nn.LayerNorm(latent_size)

    def forward(self, x, batch):
        x_dense, mask = to_dense_batch(x, batch)
        
        key_padding_mask = ~mask
        
        attn_output, _ = self.mha(
            query=x_dense, 
            key=x_dense, 
            value=x_dense, 
            key_padding_mask=key_padding_mask
        )
        x_dense = self.norm1(x_dense + attn_output)
        
        mlp_output = self.mlp(x_dense)
        x_dense = self.norm2(x_dense + mlp_output)
        
        x_sparse = x_dense[mask]
        
        return x_sparse