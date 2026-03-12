import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken

from dataset import GPTDataset
from model import GPTModel
from config import GPTConfig


def main():

    # -----------------------------
    # Device setup
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # Load text dataset
    # -----------------------------
    with open("../data/the_verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = tokenizer.encode(text)

    # -----------------------------
    # Dataset
    # -----------------------------
    config = GPTConfig()

    dataset = GPTDataset(
        token_ids=token_ids,
        max_length=config.context_length,
        stride=128
    )

    # -----------------------------
    # DataLoader
    # -----------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = GPTModel(config)
    model = model.to(device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )

    # -----------------------------
    # Loss function
    # -----------------------------
    loss_fn = nn.CrossEntropyLoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    epochs = 3

    for epoch in range(epochs):

        for step, (inputs, targets) in enumerate(dataloader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)

            # logits shape = (B,T,V)
            B, T, V = logits.shape

            # reshape for CrossEntropyLoss
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)

            # compute loss
            loss = loss_fn(logits, targets)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f}"
                )

    print("Training finished.")


if __name__ == "__main__":
    main()