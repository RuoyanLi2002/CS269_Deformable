import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling

from network import build_mlp, GraphNetBlock
from attention_network import SelfAttentionBlock

class Pool(nn.Module):
    def __init__(self, latent_size, ratio):
        super(Pool, self).__init__()
        self.pool = SAGPooling(in_channels=latent_size, ratio=ratio)

    def forward(self, x, edge_index, edge_attr, batch):
        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm, score = self.pool(x, edge_index, edge_attr, batch=batch)

        return x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm, score

class Unpool(nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    def upsample_node_features(self, x_pooled, perm, original_num_nodes):
        device = x_pooled.device
        latent = x_pooled.size(1)
        x_unpooled = torch.zeros(original_num_nodes, latent, device=device)
        x_unpooled[perm] = x_pooled
        
        return x_unpooled
    
    def forward(self, x_pooled, perm, original_num_nodes, x_encoder):
        x_unpooled = self.upsample_node_features(x_pooled, perm, original_num_nodes)

        x_unpooled = x_unpooled + x_encoder

        return x_unpooled


class GraphUnetAttention(nn.Module):
    def __init__(self, args):
        super(GraphUnetAttention, self).__init__()
        self._latent_size = args.latent_size
        self._node_input_size = args.node_input_size
        self._edge_input_size = args.edge_input_size
        self._bottom_steps = args.bottom_steps
        self._down_steps = args.down_steps
        self._up_steps = args.up_steps
        self._output_size = args.output_size
        self._ratio = args.ratio
        self._l_n = args.l_n

        self.node_embedding_mlp = build_mlp(self._node_input_size, self._latent_size, self._latent_size, lay_norm=True)
        self.edge_embedding_mlp = build_mlp(self._edge_input_size , self._latent_size, self._latent_size, lay_norm=True)

        self.bottom_processor = nn.ModuleList(SelfAttentionBlock(self._latent_size) for _ in range(self._bottom_steps))
        
        self.down_processors = nn.ModuleList()
        self.up_processors = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        for i in range(self._l_n):
            self.down_processors.append(nn.ModuleList(GraphNetBlock(self._latent_size) for _ in range(self._down_steps)))
            self.up_processors.append(nn.ModuleList(GraphNetBlock(self._latent_size) for _ in range(self._up_steps)))
            self.pools.append(Pool(self._latent_size, self._ratio))
            self.unpools.append(Unpool())

        self.decoder_mlp = build_mlp(self._latent_size , self._latent_size, self._output_size, lay_norm=False)

    def _encoder(self, x, edge_attr):
        node_latents = self.node_embedding_mlp(x)
        edge_latents = self.edge_embedding_mlp(edge_attr)
        
        return node_latents, edge_latents
    
    def _decoder(self, x):
        node_decoded = self.decoder_mlp(x)
        return node_decoded
    
    def forward(self, graph):
        node_feature = graph.x
        batch = graph.batch
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        x, edge_attr = self._encoder(x = node_feature.float(), edge_attr = edge_attr.float())

        skip_x = []
        skip_edge = []
        pool_info = []

        for i in range(self._l_n):
            for layer in self.down_processors[i]:
                x, edge_index, edge_attr = layer(x, edge_index, edge_attr)

            skip_x.append(x)
            skip_edge.append((edge_index, edge_attr, batch))
            # x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, perm, score
            x, edge_index, edge_attr, batch, perm, _ = self.pools[i](x, edge_index, edge_attr, batch)
            pool_info.append(perm)

        for layer in self.bottom_processor:
            x = layer(x, batch)

        for i in reversed(range(self._l_n)):
            perm = pool_info[i]

            original_num_nodes = skip_x[i].size(0)
            original_x = skip_x[i]
            edge_index, edge_attr, batch = skip_edge[i]
            
            x = self.unpools[i](x, perm, original_num_nodes, original_x)
            
            for layer in self.up_processors[i]:
                x, edge_index, edge_attr = layer(x, edge_index, edge_attr)
        
        decoded = self._decoder(x)

        return decoded