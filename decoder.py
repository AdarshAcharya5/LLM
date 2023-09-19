import torch.nn as nn
from feedforward import FeedForward
from multiheadattention import MultiHeadAttention

class decoder_block(nn.Module):

    def __init__(self, embed_size, num_heads, seq_length):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_size = self.embed_size // self.num_heads
        self.self_attention = MultiHeadAttention(self.num_heads, self.head_size, self.embed_size, self.seq_length)
        self.ffn = FeedForward(self.embed_size)
        self.layer_norm1 = nn.LayerNorm(self.embed_size)
        self.layer_norm2 = nn.LayerNorm(self.embed_size)

    def forward(self, x):
        y = self.self_attention(x)
        x = self.layer_norm1(x + y)
        y = self.ffn(x)
        x = self.layer_norm2(x + y)
        return x
