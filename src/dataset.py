import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - max_length, stride):

            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.inputs.append(torch.tensor(input_chunk))
            self.targets.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]