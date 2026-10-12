import logging

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedOneHotCategorical, Categorical, Bernoulli
from torch.utils.data import TensorDataset, DataLoader, random_split

# PATCH(afabench): pytorch_lightning import removed with the baseline
# models below MLP, which L2M does not use.
# import xgboost as xgb

def get_logger(logfilename):
    logger = logging.getLogger("sim_logger")
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if get_logger is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def generate_uniform_mask(batch_size, num_features, seed=None):
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        unif = torch.rand(batch_size, num_features, generator=generator)
        ref = torch.rand(batch_size, 1, generator=generator)
    else:
        unif = torch.rand(batch_size, num_features)
        ref = torch.rand(batch_size, 1)
    mask = (unif > ref).float()
    return mask

def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=-1)  # Get argmax along last dimension
    batch_size = x.shape[0]

    if len(x.shape) == 2:
        onehot = torch.zeros_like(x)
        # For each batch and sequence position, set the argmax index to 1
        onehot[torch.arange(batch_size), argmax] = 1
        return onehot
    elif len(x.shape) == 3:
        seq_len = x.shape[1]
        onehot = torch.zeros_like(x)
        # For each batch and sequence position, set the argmax index to 1
        onehot[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), argmax] = 1
        return onehot
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")

class ConcreteSelector(nn.Module):
    '''Output layer for selector models.'''

    def __init__(self):
        super().__init__()

    def forward(self, logits, temp, deterministic=False, feature_groups=None, available=None):
        if feature_groups is not None:        
            if available is not None:
                logits = logits * available

            group_logits = []
            for indices in feature_groups.values():
            
                if available is not None:
                    group_available = available[:, :, indices]  # Shape: (batch_size, seq_len, group_size)
                    all_unavailable = (group_available.sum(dim=-1) == 0)  # Shape: (batch_size, seq_len)
                    group_logits.append(torch.where(all_unavailable, 
                                                    logits[:, :, indices].sum(dim=-1) - 1e8, 
                                                    logits[:, :, indices].sum(dim=-1)))
                else:
                    group_logits.append(logits[:, :, indices].sum(dim=-1))   # Sum logits per group

            group_logits = torch.stack(group_logits, dim=-1)  # Shape: (batch_size, num_groups)
        else:
            if available is not None:
                logits = logits - 1e8 * (1 - available)
            group_logits = logits

        # If deterministic, compute the argmax. If not, sample from softmax distribution.
        if deterministic:
            probs = torch.softmax(group_logits / temp, dim=-1)
            sample = F.one_hot(probs.argmax(dim=-1), num_classes=probs.size(-1)).float()
        else:
            dist = RelaxedOneHotCategorical(temp, logits=group_logits)
            soft_sample = dist.rsample()

            hard_sample = torch.zeros_like(soft_sample)
            hard_sample.scatter_(-1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
            sample = hard_sample.detach() - soft_sample.detach() + soft_sample

        if feature_groups is not None:
            batch_size, seq_len, _ = logits.shape
            num_features = logits.size(-1)
            mapped_sample = torch.zeros((batch_size, seq_len, num_features), device=logits.device)

            for group_idx, indices in enumerate(feature_groups.values()):
                mapped_sample[:, :, indices] = sample[:, :, group_idx].unsqueeze(-1)

            return mapped_sample  # (B, T, F)
        else:
            return sample 
        
def MLP(input_dim, output_dim, hidden_dim, num_layers, policy_head=False, dropout=None):
    layers = []
    if num_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))

    if policy_head:
        nn.init.orthogonal_(layers[-1].weight, gain=0.01)
        nn.init.constant_(layers[-1].bias, 0.0)

    return nn.Sequential(*layers)

# PATCH(afabench): BaselineNN, BaselineNNLightning and the baseline
# training helpers that followed are removed; L2M does not use them.
