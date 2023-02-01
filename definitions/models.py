
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, block_size=8, head_size=32):
        super().__init__()
        self.q = torch.nn.Linear(embed_dim, head_size, bias=False)
        self.k = torch.nn.Linear(embed_dim, head_size, bias=False)
        self.v = torch.nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril_mask',
                 torch.tril(torch.ones(block_size, block_size)))


    def forward(self, tokens):
        batch, block, embed_dim = tokens.shape
        # init token projection
        k = self.k(tokens)
        q = self.q(tokens)
        # query key attention; normalize by sqrt(d_k)
        wei = torch.matmul(q, k.view(batch, embed_dim, block)).pow(-0.5)
        wei = wei.masked_fill(self.tril_mask == 0, -1e9).softmax(dim=-1)
        # weighted aggregation
        v = self.v(tokens)
        output = torch.matmul(wei, v)
        return output


class BiGram(nn.Module):
    
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn_head = AttentionHead(embed_dim)
        self.linear_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens, target=None):
        batch, block_size = tokens.shape

        # (batch, block, vocab_size)
        token_embed = self.embedding(tokens)
        # (batch, block)
        pos_embed = self.pos_embedding(torch.arange(block_size))
        x = token_embed + pos_embed

        x = self.attn_head(x)
        logits = self.linear_head(x)
        
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

    def generate(self, idx=None, block_size=8, max_new_tokens=100):

        if idx is None:
            idx = torch.zeros((1, 1), dtype=torch.long)

        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -block_size:]
            # (batch, block, vocab_size)
            logits, _ = self.forward(idx)
            # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]
            # (1, 1, vocab_size)
            probs = logits.softmax(dim=1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx
           