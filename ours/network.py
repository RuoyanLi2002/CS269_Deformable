import torch
import torch.nn as nn
from torch_scatter import scatter_add

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(nn.Linear(in_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Linear(hidden_size, out_size))
    if lay_norm:
        return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    
    return module

class GraphNetBlock(nn.Module):
    def __init__(self, hidden_size):
        super(GraphNetBlock, self).__init__()
        
        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size

        self.edge_mlp = build_mlp(eb_input_dim, hidden_size, hidden_size)
        self.node_mlp = build_mlp(nb_input_dim, hidden_size, hidden_size)
    
    def edge_update(self, node_features, edge_index, edge_attr):
        # Edge update
        senders_idx, receivers_idx = edge_index
        senders_attr = node_features[senders_idx]
        receivers_attr = node_features[receivers_idx]
        collected_edges = torch.cat([senders_attr, receivers_attr, edge_attr], dim=-1)
        updated_edge_attr = self.edge_mlp(collected_edges)

        return updated_edge_attr
    
    def node_update(self, node_features, edge_index, updated_edge_attr):
        # Node update
        senders_idx, receivers_idx = edge_index

        aggregated_edges = scatter_add(updated_edge_attr, receivers_idx, dim=0, dim_size=node_features.size(0))
        collected_nodes = torch.cat([node_features, aggregated_edges], dim=-1)
        updated_node_features = self.node_mlp(collected_nodes)

        return updated_node_features

    def forward(self, node_features, edge_index, edge_attr):
        """
        node_features (Tensor): Node feature matrix of shape [num_nodes, node_feature_dim].
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        edge_attr (Tensor): Edge feature matrix of shape [num_edges, edge_feature_dim].
        """
        original_node_features = node_features.clone()
        original_edge_attr = edge_attr.clone()

        updated_edge_attr = self.edge_update(node_features, edge_index, edge_attr)
        updated_node_features = self.node_update(node_features, edge_index, updated_edge_attr)

        # Add residual connections
        node_features = original_node_features + updated_node_features
        edge_attr = original_edge_attr + updated_edge_attr

        return node_features, edge_index, edge_attr