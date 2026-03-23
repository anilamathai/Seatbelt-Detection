import torch
import yaml
import os

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
