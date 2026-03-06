import torch 
from torch.utils.data import Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.inputs = []
        self.targets = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            self.inputs.append(torch.tensor(token_ids[i:i + max_length]))
            self.targets.append(torch.tensor(token_ids[i + 1:i + 1 + max_length]))
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return [self.inputs[idx], self.targets[idx]]