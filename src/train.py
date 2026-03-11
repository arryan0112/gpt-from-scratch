import torch
from torch.utils.data import DataLoader
import tiktoken

from dataset import GPTDataset


with open("../data/the_verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = tiktoken.get_encoding("gpt2")

token_ids = tokenizer.encode(text)


dataset = GPTDataset(
    token_ids=token_ids,
    max_length=256,
    stride=128
)


dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)