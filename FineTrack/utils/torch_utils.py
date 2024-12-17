import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def assert_accelerator():
    assert torch.cuda.is_available() or torch.backends.mps.is_available(), "Hardware acceleration is mandatory, but only no device was found."

def get_device_str():
    return str(get_device())

def gradients_norm(module: torch.nn.Module):
    total_norm = 0
    for p in module.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
