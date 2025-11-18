# networks.py
import torch
import torch.nn as nn

class ConcatMlp(nn.Module):
    """连接输入特征和动作的多层感知机"""
    
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_size = input_dim
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            prev_size = size
        self.layers.append(nn.Linear(prev_size, output_dim))
    
    def forward(self, x, a=None):
        if a is not None:
            x = torch.cat([x, a], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x