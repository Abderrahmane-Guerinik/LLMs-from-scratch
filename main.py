import tiktoken
from datasets.create_dataloader import create_dataloader_v1

enc = tiktoken.encoding_for_model("gpt-2")

with open('data/the-verdict.txt', encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=1,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, outputs  = next(data_iter)

print(inputs)
print('\n')
print(outputs)