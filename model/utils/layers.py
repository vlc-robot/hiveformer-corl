import torch.nn as nn
import torch.nn.functional as F

from .multihead_custom_attention import MultiheadCustomAttention


class CrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, value, query_pos=None, value_pos=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query=(query + query_pos) if query_pos is not None else query,
            key=(value + value_pos) if value_pos is not None else value,
            value=value
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights


class RelativeCrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, query_pos=None, value_pos=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query=query,
            key=value,
            value=value,
            rotary_pe=(query_pos, value_pos) if query_pos is not None else None
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)


class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output
