
# %% imports
import os
import torch
import tiktoken as tk

from definitions.models import Bigram
from utils.config import *

# %% hyperparameters
PATH = os.path.join('data', "raw-1.txt")
with open(PATH, 'r') as f:
    text = f.read()


# %% preprocessing

# %%
num_chars = 100_000
encoder = tk.get_encoding('gpt2')
data = torch.tensor(encoder.encode(text[:num_chars]), dtype=torch.long)

num_train_samples = int(0.9 * len(data))
train_data = data[:num_train_samples]
val_data = data[num_train_samples:]

# %%
block_size = 8
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    rand_idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in rand_idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in rand_idx])
    return x, y

print(get_batch('train'))

# %%
model = Bigram(encoder.vocab_size)

x, y = get_batch('train')
print(model(x))


# %%
