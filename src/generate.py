import torch
import tiktoken

from config import GPTConfig
from load_gpt2_weights import load_weights


def generate(model, idx, max_new_tokens, context_length):

    for _ in range(max_new_tokens):

        # crop context if needed
        idx_cond = idx[:, -context_length:]

        # forward pass
        logits = model(idx_cond)

        # take logits from last token
        logits = logits[:, -1, :]

        # convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # append token
        idx = torch.cat((idx, next_token), dim=1)

    return idx


def main():

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Using device:", device)

    # load GPT-2 weights into your architecture
    model = load_weights()

    model = model.to(device)

    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "The courtroom was silent"

    input_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    output = generate(
        model,
        input_ids,
        max_new_tokens=50,
        context_length=1024
    )

    generated_text = tokenizer.decode(output[0].tolist())

    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()