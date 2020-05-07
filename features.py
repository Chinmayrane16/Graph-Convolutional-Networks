import torch

def one_hot_features(dims):
    return torch.eye(dims)