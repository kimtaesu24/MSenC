import random
import torch
import numpy as np
from loguru import logger

def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
def log_args(args):
    for arg_name, arg_value in vars(args).items():
        if type(arg_value) is dict:
            for in_key, in_value in arg_value.items():
                logger.info('{:20}:{:>30}'.format(in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>30}'.format(arg_name, '{}'.format(arg_value)))


def get_model_info(model, layer=False):
    # Print information about each layer
    if layer:
        print("\nLayer Information:")
        for name, param in model.named_parameters():
            print(f"Layer: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    
    # Count total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    
def dict_to_gpu(my_dict, device):
    return {key: value.to(device) for key, value in my_dict.items()}