import random
import argparse
import numpy as np
from contextlib import contextmanager

import torch


def str2bool(input_):
    if input_.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_model_value(model):
    params = list(model.named_parameters())
    print (params[-1][0],params[-1][1][:4])
    
    
def random_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
# helpers for performer generator

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

def prepare_inputs_for_generation(self, input_ids, **kwargs):
    return {"input_ids": input_ids}
    
def adjust_logits_during_generation(self, logits, **kwargs):
    return logits


      