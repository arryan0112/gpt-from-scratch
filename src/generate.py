import torch
import tiktoken

from config import GPTConfig
from model import GPTModel


def generate(model, idx, max_new_tokens, context_length):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_length:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx


def main():

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    config = GPTConfig()

    model = GPTModel(config)

    model.load_state_dict(torch.load("gpt_model.pth", map_location=device))

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
        context_length=config.context_length
    )

    generated_text = tokenizer.decode(output[0].tolist())

    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()