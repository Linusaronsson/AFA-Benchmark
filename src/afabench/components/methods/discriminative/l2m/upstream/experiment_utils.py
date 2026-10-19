import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# PATCH(afabench): import path of the vendored copy.
from afabench.components.methods.discriminative.l2m.upstream.utils import ConcreteSelector

def get_eval_dir(config):
    return os.path.join(config['eval_data_dir'], f'{config["experiment"]}_{config["num_points"]}')

def get_eval_path(config, task):
    path = get_eval_dir(config)
    filename = f'{config["eval_kernel"]}-seed{config["eval_seed"]}-{task}'
    filename += '.tar'
    return path, filename

def gen_evalset(config, sampler, task='afa'):
    print(f"Generating Evaluation Sets for {config['experiment']}")
    # Sample full batch
    if config['experiment'] == 'sim':
        batch = sampler.sample(
            batch_size=config['eval_batch_size'],
            max_num_points=config['num_points'],
            device=config['device'],  # PATCH(afabench): the fit's device, not 'cuda'
            x_dim=config['feature_dim'],
            num_observed=config['feature_dim'],
            seed=config['train_seed'])
    else:
        batch = sampler.sample(
            batch_size=config['eval_batch_size'],
            max_num_points=config['num_points'],
            device=config['device'],  # PATCH(afabench): the fit's device, not 'cuda'
            num_observed=config['feature_dim'],
            seed=config['train_seed'])

    eval_batch = sampler.mask_features(batch)

    batch_dict = {
        'full': batch,
        'eval': eval_batch,
    }

    # PATCH(afabench): returned, not cached under eval_data_dir; the fit
    # generates its validation tasks once in memory.
    return batch_dict

def eval(config, sampler, model, accelerator):
    # Evaluate on a predefined saved set of evaluation samples
    # PATCH(afabench): read from the config, where the fit stores the
    # output of gen_evalset, instead of a file under eval_data_dir.
    batch_dict = config['eval_sets']['predictor']
    batch = batch_dict['eval']
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        # Move batch tensors to the correct device
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outs = model(batch)
        total_loss += outs['loss'].detach().item()
    line = f"[EVAL] avg_loss: {total_loss:.6f} over {config['eval_batch_size']} batches"
    return total_loss, line

def eval_afa(config, sampler, model, accelerator, temp):
    # Evaluate on a predefined saved set of evaluation samples
    # PATCH(afabench): read from the config, where the fit stores the
    # output of gen_evalset, instead of a file under eval_data_dir.
    batch_dict = config['eval_sets']['afa']
    eval_batch = batch_dict['eval']
    full_batch = batch_dict['full']

    collapse_stats = {"entropy": 0.0, "mode_frac": 0.0}
    eps = 1e-8

    model.eval()
    selector_fn = ConcreteSelector()
    total_loss = 0.0
    with torch.no_grad():
        # Move batch tensors to the correct device
        eval_batch = {k: v.to(accelerator.device) for k, v in eval_batch.items()}
        # TODO: enable support for feature groups
        m = eval_batch['maskt']

        outs = model.select_action(eval_batch)
        available = (1 - m) * eval_batch['rt']
        has_available = (available.sum(dim=-1) > 0).float().unsqueeze(-1)

        # Compute collapse statistics
        #probs = selector_fn(outs, temp=temp, feature_groups=sampler.feature_groups, available=available)
        probs = outs.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1).mean().item()
        action_preds = outs.argmax(dim=-1).view(-1)
        counts = torch.bincount(action_preds, minlength=outs.size(-1))
        mode_frac = (counts.max().float() / action_preds.numel()).item()

        collapse_stats["entropy"] += entropy
        collapse_stats["mode_frac"] += mode_frac 

        khot = selector_fn(outs, temp=temp, deterministic=True, feature_groups=sampler.feature_groups, available=available)
        m = torch.max(m, khot)
        
        batch_soft = sampler.mask_features(full_batch, mask=m)  
        outs = model(batch_soft, reduce_ll=False)

        loss = (outs['loss'] * has_available).sum() / (has_available.sum() + 1e-8)
        total_loss += loss

    line = f"[EVAL] avg_loss: {total_loss:.6f} over {config['eval_batch_size']} batches, collapse_stats: {collapse_stats}"
    return total_loss, line