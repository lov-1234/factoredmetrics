import torch

def log_gaussian(x, mean, logvar):
    return -0.5 * (logvar + (x - mean) ** 2 / torch.exp(logvar))
