
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bigram(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens, target=None):
        # (batch, block, vocab_size)
        logits = self.embedding(tokens)
        
        # (batch*block, vocab_size)
        if target is None:
            loss = None
        else:
            # reshape dims
            batch, block, vocab_size = logits.shape
            logits = logits.view(batch*block, vocab_size)
            target = target.view(-1)
            loss = F.cross_entropy(logits, target)
            
        return logits, loss

    def generate(self, idx=None, max_new_tokens=100):

        if idx is None:
            idx = torch.zeros((1, 1), dtype=torch.long)

        for _ in range(max_new_tokens):
            # (batch, block, vocab_size)
            logits, _ = self.forward(idx)
            # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]
            # (1, 1, vocab_size)
            probs = logits.softmax(dim=1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx
           