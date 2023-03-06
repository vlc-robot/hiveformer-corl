import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class RelativeCrossAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers):
        super().__init__()

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(embedding_dim, num_attn_heads))
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, query, value, query_pos=None, value_pos=None):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](query, value, query_pos, value_pos)
            query = self.ffw_layers[i](query)
            output.append(query)
        return output


class TaskSpecificRelativeCrossAttentionLayer(nn.Module):
    """Relative cross attention layer with task specific biases."""
    def __init__(self, embedding_dim, num_heads, tasks, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.task_biases = nn.ParameterDict()
        for task in tasks:
            self.task_biases[f"{task}_multihead_attn_in_proj_bias"] = nn.Parameter(
                torch.zeros_like(self.multihead_attn.in_proj_bias))
            self.task_biases[f"{task}_multihead_attn_out_proj_bias"] = nn.Parameter(
                torch.zeros_like(self.multihead_attn.out_proj.bias))
            self.task_biases[f"{task}_norm_bias"] = nn.Parameter(
                torch.zeros_like(self.norm.bias))

    def forward(self, task, query, value, query_pos=None, value_pos=None):
        output = torch.zeros_like(query)

        for t in np.unique(task):
            self.multihead_attn.in_proj_bias = self.task_biases[f"{t}_multihead_attn_in_proj_bias"]
            self.multihead_attn.out_proj.bias = self.task_biases[f"{t}_multihead_attn_out_proj_bias"]
            self.norm.bias = self.task_biases[f"{t}_norm_bias"]

            query_task = query[:, task == t]
            value_task = value[:, task == t]
            if query_pos is not None:
                query_pos_task = query_pos[task == t]
                value_pos_task = value_pos[task == t]
            attn_output_task, attn_output_weights = self.multihead_attn(
                query=query_task,
                key=value_task,
                value=value_task,
                rotary_pe=(query_pos_task, value_pos_task) if query_pos is not None else None
            )
            output_task = query_task + self.dropout(attn_output_task)
            output_task = self.norm(output_task)
            output[:, task == t] = output_task

        return output, None


class TaskSpecificFeedforwardLayer(nn.Module):
    """Feedforward layer with task specific biases."""
    def __init__(self, embedding_dim, hidden_dim, tasks, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

        self.task_biases = nn.ParameterDict()
        for task in tasks:
            self.task_biases[f"{task}_linear1_bias"] = nn.Parameter(
                torch.zeros_like(self.linear1.bias))
            self.task_biases[f"{task}_linear2_bias"] = nn.Parameter(
                torch.zeros_like(self.linear2.bias))
            self.task_biases[f"{task}_norm_bias"] = nn.Parameter(
                torch.zeros_like(self.norm.bias))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, task, x):
        output = torch.zeros_like(x)

        for t in np.unique(task):
            self.linear1.bias = self.task_biases[f"{t}_linear1_bias"]
            self.linear2.bias = self.task_biases[f"{t}_linear2_bias"]
            self.norm.bias = self.task_biases[f"{t}_norm_bias"]

            x_task = x[:, task == t]
            output_task = self.linear2(self.dropout(self.activation(self.linear1(x_task))))
            output_task = x_task + self.dropout(output_task)
            output_task = self.norm(output_task)
            output[:, task == t] = output_task

        return output


class TaskSpecificRelativeCrossAttentionModule(nn.Module):
    """Relative cross attention module with task specific biases."""
    def __init__(self, embedding_dim, num_attn_heads, num_layers, tasks):
        super().__init__()

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(TaskSpecificRelativeCrossAttentionLayer(embedding_dim, num_attn_heads, tasks))
            self.ffw_layers.append(TaskSpecificFeedforwardLayer(embedding_dim, embedding_dim, tasks))

    def forward(self, task, query, value, query_pos=None, value_pos=None):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](task, query, value, query_pos, value_pos)
            query = self.ffw_layers[i](task, query)
            output.append(query)
        return output
