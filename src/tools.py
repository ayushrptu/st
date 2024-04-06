import numpy as np
from functools import partial
import torch
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Miscellaneous tools


# Count elements on GPU
def count_gpu():
    L = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and obj.device == "cuda:0":
                L += obj.numel()
        except:
            pass
    return L


# Make metric compatible with VAE Net Output
def vae_compatible(metric_):
    def metric_wrapper(outputs, labels, **kwargs):
        # ignore extra output
        if type(outputs) == tuple:
            outputs, kl = outputs
        return metric_(outputs, labels, **kwargs)
    return metric_wrapper


# Make Metric.__call__ compatible with VAE Net Output
def vae_compatible_call(metric_):
    def call_wrapper(self, outputs, labels, **kwargs):
        # ignore extra output
        if type(outputs) == tuple:
            outputs, kl = outputs
        return metric_(self, outputs, labels, **kwargs)
    return call_wrapper


# To stop a forward pass before completion
class StopForward(Exception):
    pass


# Counted access
class CountedDict(dict):
    def __init__(self, pre_dict):
        self.counts = dict.fromkeys(pre_dict.keys(), 0)
        super().__init__()
        super().update(**pre_dict)

    def __getitem__(self, key):
        if key in self.counts.keys():
            self.counts[key] += 1
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.counts.keys():
            self.counts[key] += 1
        return super().get(key, default)


# Temporarily freeze net with context manager
class Freeze:
    def __init__(self, model):
        """ freeze this models parameters by setting requires_grad to false """
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]

    def __enter__(self):
        for p in self.params:
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.params:
            p.requires_grad = True


# Rescale weights in combined loss such that sum(weights) = len(weights)
def rescale(weights):
    weights = np.array(weights)
    return len(weights) * weights / sum(weights)


# Infinity iterable dataset
class InfIter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.iter = iter(dataset)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataset)
            return next(self.iter)


# Wrap Lambda as Module for Sequential
class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# Access model layers
def get_act(model, batch, hooks, last_hook=None):
    """
    Get activation after specified module
    """
    if type(hooks) not in [list, tuple]:
        hooks = [hooks]
    feats = {}

    # Attach hooks for feature activations
    def save_hook(m, inp, op, feats, m_name, stop=False):
        feats[m_name] = op
        if stop:
            raise StopForward

    handles = []
    for m in hooks:
        hk = dict(model.named_modules())[m].register_forward_hook(
            partial(save_hook, feats=feats, m_name=m, stop=m == last_hook)
        )
        handles += [hk]

    # Forward pass
    with Freeze(model):
        with torch.no_grad():
            try:
                b1 = batch["x"].to(device)
                out = model(b1.to(device), src_mask=torch.zeros_like(b1[:, :, 0]) == 1)
                feats["out"] = out[0]
            except StopForward:
                pass

    # Remove hooks
    for hk in handles:
        hk.remove()

    if len(feats) == 1:
        return feats.values()[0]
    return feats
