import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import decoder_block

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_length, embed_size, num_layers, num_heads):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_embedding = nn.Embedding(self.seq_length, self.embed_size)
        self.decoder_blocks = nn.Sequential(*[decoder_block(self.embed_size, self.num_heads, self.seq_length) for _ in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.embed_size)
        self.linear = nn.Linear(self.embed_size, self.vocab_size)
        self.apply(self.init__weights)

    def init__weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


    def forward(self, context, targets=None):
        emb = self.embedding(context)
        pos_emb = self.positional_embedding(torch.arange(self.seq_length, device=self.device))
        x = emb + pos_emb
        x = self.decoder_blocks(x)
        x = self.layer_norm(x)
        logits = self.linear(x)
        if targets==None:
            loss = None
        else:
            batch_size, seq_length, vocab_size = logits.shape
            logits = logits.view(batch_size * seq_length, vocab_size)
            targets = targets.view(batch_size * seq_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def sample(self, context, max_tokens = 500):
        for _ in range(max_tokens):
            logits, loss = self.forward(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=-1)
        return context
