import tiktoken
import torch
import torch.nn as nn
from datasets.create_dataloader import create_dataloader_v1

enc = tiktoken.encoding_for_model("gpt-2")

with open('data/the-verdict.txt', encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, outputs  = next(data_iter)

print(inputs)
print('\n')
print(outputs)

vocab_size = 50257
output_dim = 256
context_length = 4

embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = nn.Embedding(4, output_dim)

token_embeddings = embedding_layer(inputs)

pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings


