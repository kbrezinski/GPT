
import torch

from utils.constants import *


class BiGramTrainer():

    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, train_data, val_data):

        # start training loop
        for epoch in range(num_epochs):
            
            # evaluation step
            if not epoch % eval_iters:
                losses = self.evaluate(val_data)
                print(f'Epoch: {epoch:4d}, Loss: {losses:.4f}')

            # training step
            self.optimizer.zero_grad(set_to_none=True)
            x, y = self.get_batch(train_data)
            _, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()

    # fetches a batch of data randomly from the split data
    def get_batch(self, data):
        rand_idx = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in rand_idx])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in rand_idx])
        return x.to(self.device), y.to(self.device)

    # evaluation step on the validation data
    @torch.no_grad()
    def evaluate(self, val_data):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for iters in range(eval_iters):
            x, y = self.get_batch(val_data)
            _, loss = self.model(x, y)
            losses[iters] = loss.item()
        self.model.train()
        return losses.mean()

    # generate new text
    def generate(self, idx=None, max_new_tokens=100):
        return self.model.generate(idx, max_new_tokens)


class Tokenizer():
    def __init__(self):
        pass

    def train(self, file_dir='data/raw-1.txt'):
        # read in file
        with open(file_dir, 'r') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # custom encoder decoder
        string_to_int = {c: i for i, c in enumerate(chars)}
        int_to_string = {i: c for i, c in enumerate(chars)}

        self.encoder = lambda string: [string_to_int[c] for c in string]
        self.decoder = lambda ints: ''.join([int_to_string[i] for i in ints])

        data = torch.tensor(self.encoder(text), dtype=torch.long)

        num_train_samples = int(train_frac * len(data))
        train_data = data[:num_train_samples]
        val_data = data[num_train_samples:]

        return train_data, val_data