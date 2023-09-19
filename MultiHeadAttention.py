import torch
import torch.nn as nn
from SelfAttention import SelfAttention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, embed_size, seq_length, dropout_=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.embed_size = embed_size
        self.seq_length = seq_length
        self.heads = nn.ModuleList([SelfAttention(self.head_size, self.embed_size, self.seq_length) for _ in range(self.num_heads)])
        self.linear = nn.Linear(self.num_heads * self.head_size, self.embed_size)
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.dropout(self.linear(x))
        return x
