
import torch
import torch.nn as nn

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super(Bigram).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        logits = self.embedding(idx)
        return logits