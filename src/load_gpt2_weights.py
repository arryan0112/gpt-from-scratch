import torch
from transformers import GPT2LMHeadModel
from config import GPTConfig
from model import GPTModel


def load_weights():

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading GPT-2 weights...")

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    hf_model.eval()

    config = GPTConfig()

    model = GPTModel(config)

    sd = model.state_dict()

    hf_sd = hf_model.state_dict()

    # embeddings
    sd["embedding.token_emb.weight"] = hf_sd["transformer.wte.weight"]
    sd["embedding.pos_emb.weight"] = hf_sd["transformer.wpe.weight"]

    for i in range(config.n_layers):

        # combined qkv weight
        w = hf_sd[f"transformer.h.{i}.attn.c_attn.weight"]
        b = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"]

        w = w.T

        q, k, v = w.split(config.emb_dim, dim=0)
        qb, kb, vb = b.split(config.emb_dim)

        sd[f"blocks.{i}.attn.q_proj.weight"] = q
        sd[f"blocks.{i}.attn.k_proj.weight"] = k
        sd[f"blocks.{i}.attn.v_proj.weight"] = v

        sd[f"blocks.{i}.attn.q_proj.bias"] = qb
        sd[f"blocks.{i}.attn.k_proj.bias"] = kb
        sd[f"blocks.{i}.attn.v_proj.bias"] = vb

        # attention output
        sd[f"blocks.{i}.attn.out_proj.weight"] = hf_sd[f"transformer.h.{i}.attn.c_proj.weight"].T
        sd[f"blocks.{i}.attn.out_proj.bias"] = hf_sd[f"transformer.h.{i}.attn.c_proj.bias"]

        # layer norms
        sd[f"blocks.{i}.ln1.weight"] = hf_sd[f"transformer.h.{i}.ln_1.weight"]
        sd[f"blocks.{i}.ln1.bias"] = hf_sd[f"transformer.h.{i}.ln_1.bias"]

        sd[f"blocks.{i}.ln2.weight"] = hf_sd[f"transformer.h.{i}.ln_2.weight"]
        sd[f"blocks.{i}.ln2.bias"] = hf_sd[f"transformer.h.{i}.ln_2.bias"]

        # MLP
        sd[f"blocks.{i}.ff.net.0.weight"] = hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"].T
        sd[f"blocks.{i}.ff.net.0.bias"] = hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"]

        sd[f"blocks.{i}.ff.net.2.weight"] = hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"].T
        sd[f"blocks.{i}.ff.net.2.bias"] = hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"]

    # final layer norm
    sd["ln_f.weight"] = hf_sd["transformer.ln_f.weight"]
    sd["ln_f.bias"] = hf_sd["transformer.ln_f.bias"]

    # output head
    sd["head.weight"] = hf_sd["lm_head.weight"]

    model.load_state_dict(sd)

    model = model.to(device)

    print("GPT-2 weights loaded successfully!")

    return model