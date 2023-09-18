import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, head_size, embed_size, seq_length, dropout_=0.2):
        super().__init__()
        self.head_size = head_size
        self.embed_size = embed_size
        self.seq_length = seq_length
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(self.seq_length, self.seq_length)))
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = (q @ k.transpose(-2, -1)) * k.shape[-1]**-0.5  # q @ k^T / sqrt(d)
        att = att.masked_fill(self.mask[:seq_length, :seq_length] == 0, float('-inf')) # replace 0 with -inf so softmax doesn't crash kekw
        att = self.dropout(F.softmax(att, dim=-1))
        att = att @ v
        return att


