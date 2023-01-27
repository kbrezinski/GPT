
# mini-GPT

![GitHub](https://img.shields.io/github/license/kbrezinski/mini-GPT?logo=MIT)

This is a minimal implementation of GPT-2 in PyTorch. It is intended to be a starting point for researchers and students to understand the model and experiment with it. It is not intended to be a production-ready implementation. 

Some main features of the package are the following:
1. Uses the same vocabulary as the original GPT-2 model based on the `tiktoken` tokenizer which is a BPE tokenizer that is much faster than Hugginface implementations as noted in the [tiktoken repo](https://github.com/openai/tiktoken).