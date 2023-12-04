import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))


class GATBlock(nn.Module):
    def __init__(self, emb_size, ll_hidden_size, att_heads, edge_fea_dim):
        super(GATBlock, self).__init__()
        self.ffn = FFN(emb_size, ll_hidden_size)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = F.dropout
        self.conv = GATv2Conv(
            in_channels=emb_size,
            out_channels=emb_size // att_heads,
            heads=att_heads,
            concat=True,
            edge_dim=edge_fea_dim
        )

    def forward(self, h, edge_index, edge_attr=None):
        u = self.ffn(h)
        v = self.layer_norm(h + self.dropout(u))
        w, weights = self.conv(v, edge_index, edge_attr, return_attention_weights=True)
        output = self.layer_norm(self.dropout(w) + v)
        return output, weights


class GraphAttentionNetwork(nn.Module):
    def __init__(self, num_layers, emb_size, ll_hidden_size, att_heads, edge_fea_dim):
        super(GraphAttentionNetwork, self).__init__()

        self.GATBlocks = []
        self.num_layers = num_layers
        for i in range(num_layers):
            self.GATBlocks.append(GATBlock(emb_size, ll_hidden_size, att_heads, edge_fea_dim))

    def forward(self, h, edge_index, edge_attr=None):
        x = h
        outputs = []
        for i in range(self.num_layers):
            x, weights = self.GATBlocks[i](x, edge_index, edge_attr)
            outputs.append((x, weights))
        return outputs
