import torch.nn as nn

class feedforward(nn.Module):

    def __init__(self, embed_size, dropout_=0.2):
        super().__init__()
        self.embed_size = embed_size
        self.dropout_ = dropout_
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size * 4),
            nn.ReLU(),
            nn.Linear(self.embed_size * 4, self.embed_size),
            nn.Dropout(self.dropout_),
        )

    def forward(self, x):
        return self.fc(x)
