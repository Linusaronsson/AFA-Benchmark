# PATCH(afabench): pandas and polars imports removed with the loaders below.
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# PATCH(afabench): normalize_icd10, extract_demo and preprocess_df are
# removed; they load the authors' datasets, not the task prior.

class TabularDataset(Dataset):
    def __init__(self, data, dim_y, feature_names, dataframe, free_indices, label=None, mask=None, feature_groups=None):
        self.data = data
        self.dim_y = dim_y
        self.feature_names = feature_names
        self.dataframe = dataframe
        self.free_indices = free_indices
        self.feature_dim = data.shape[-1]
        self.feature_groups = feature_groups
        self.label = label
        self.mask = mask

        if feature_groups is None:
            self.num_available_features = self.feature_dim - len(free_indices)
        else:
            self.num_available_features = len(feature_groups)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_dataframe(self):
        return self.dataframe
    
def sample_random_function(
    x: torch.Tensor,
    output_dim: int = 1,
    min_feats: int = 1,
    max_feats: int = None,
    hidden_dim: int = 8,
    max_prob: float = 1.0,
    informative_features: list[int] = None,
    target_prevalence_range: tuple = (0.05, 0.95),
    temperature_range: tuple = (0.5, 1.5),
    subgroups: int = None,
    return_groups: bool = False,
    seed: int = None,
    label: bool = False,
    return_p: bool = False,
) -> torch.Tensor:
    """
    x: [B, N, D]
    min_feats, max_feats: inclusive bounds on # informative features
    hidden_dim: width of the random hidden layer
    """
    B, N, D = x.shape

    if seed is not None:
        generator = torch.Generator(device=x.device).manual_seed(seed)
    else:
        # PATCH(afabench): torch.seed() reseeds the global generator
        # nondeterministically, so a fit could not be reproduced from its
        # seed; draw the generator's seed from the global generator instead.
        generator = torch.Generator(device=x.device).manual_seed(int(torch.randint(2**62, (1,)).item()))
    
    groups = []
    batch_feature_indices = []

    # 1) For each batch element, sample K_i in [min_feats, max_feats]
    if informative_features is None:
        # Generate random number of clusters (1-3) for each batch
        if subgroups is None:
            num_clusters = torch.randint(1, 4, (B,), device=x.device, generator=generator)
        else:
            num_clusters = torch.full((B,), subgroups, device=x.device)
        cluster_centers = torch.randn(B, num_clusters.max(), D, device=x.device, generator=generator)
        
        feat_masks = torch.zeros(B, N, D, device=x.device)
        for b in range(B):
            # [N, D] - current sample
            x_b = x[b]
            centers_b = cluster_centers[b, :num_clusters[b]]  # [K_b, D]
            dists = torch.cdist(x_b.unsqueeze(0), centers_b.unsqueeze(0)).squeeze(0)
            cluster_ids = dists.argmin(dim=1)  # [N]
            groups.append(cluster_ids)
            feature_indices = {}

            for c in range(num_clusters[b]):
                cluster_mask = (cluster_ids == c).nonzero(as_tuple=False).squeeze(1)
                if max_feats is not None:
                    K_c = torch.randint(min_feats, max_feats + 1, (1,), device=x.device, generator=generator)
                else:
                    K_c = torch.randint(min_feats, D + 1, (1,), device=x.device, generator=generator)
                idx_c = torch.randperm(D, device=x.device, generator=generator)[:K_c]
                feature_indices[c] = idx_c
                if len(cluster_mask) > 0:
                    rows, cols = torch.meshgrid(cluster_mask, idx_c, indexing="ij")  # [N_c, K_c]
                    feat_masks[b, rows, cols] = 1.0
            
            batch_feature_indices.append(feature_indices)

    else:
        feat_masks = torch.zeros(B, N, D, device=x.device)
        feat_masks[:, :, informative_features] = 1.0

    # Now feat_masks[b, d] == 1 iff feature d is “informative” for batch‐row b

    # 3) Apply that mask to every time‐step / row: [B, N, D]
    # if output_dim == 1:
    #     print(batch_feature_indices[0:3])
    #     print(feat_masks[0:3,0,:])
    x_inf = x * feat_masks

    # 4) Sample random NN weights
    scale1 = 0.1 + 1.4 * torch.rand(B, 1, 1, device=x.device, generator=generator)
    scale2 = 0.1 + 1.4 * torch.rand(B, 1, 1, device=x.device, generator=generator)
    feature_importance = torch.empty(B, D, 1, device=x.device)
    feature_importance.uniform_(1, 10, generator=generator)

    w1 = torch.randn(B, D, hidden_dim, device=x.device, generator=generator) * feature_importance * scale1
    b1 = torch.randn(B, 1, hidden_dim, device=x.device, generator=generator) * scale1
    w2 = torch.randn(B, hidden_dim, output_dim, device=x.device, generator=generator) * scale2
    b2 = torch.randn(B, 1, output_dim, device=x.device, generator=generator) * scale2

    # 5) Forward pass through the per‐batch random NN
    #    note: bmm(x_inf, w1) does ∑_d x_inf * w1[d→h]  → [B, N, H]
    h = torch.tanh(torch.bmm(x_inf, w1) + b1)    # [B, N, H]
    y_mean = torch.bmm(h, w2) + b2               # [B, N, 1]

    logits = y_mean
    temperatures = torch.empty(B, 1, 1, device=x.device).uniform_(*temperature_range, generator=generator)
    logits = logits / temperatures

    if label and output_dim == 1:
        target_prevalences = torch.empty(B, 1, 1, device=x.device).uniform_(*target_prevalence_range, generator=generator)
        bias_shift = torch.log(target_prevalences / (1 - target_prevalences))
        logits = logits + bias_shift 
    elif label and output_dim > 1:
        K = logits.shape[-1]
        c_min, c_max = 0.2, 2
        c = torch.empty(B, 1, device=logits.device).uniform_(c_min, c_max, generator=generator)
        alpha = c.expand(B, K)    
        if seed is not None:
            # PATCH(afabench): fork every visible CUDA device's generator,
            # none without CUDA. devices=[x.device] forks the current CUDA
            # device for a CPU x, and fails where CUDA is unavailable.
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                pi = torch.distributions.Dirichlet(alpha).sample()
        else:
            pi = torch.distributions.Dirichlet(alpha).sample()
        bias_shift = pi.clamp_min(1e-8).log().view(B, 1, K)  
        logits = logits + bias_shift

    # 7) Sigmoid + Bernoulli
    probs = torch.sigmoid(logits) * max_prob

    if label and output_dim > 1:
        flat_probs = probs.reshape(-1, output_dim)
        idx = torch.multinomial(flat_probs, 1, generator=generator)  # [B*T, 1]
        y = torch.nn.functional.one_hot(idx.squeeze(-1), num_classes=output_dim).float()
        y = y.reshape(B, N, output_dim)
    else:
        y = torch.bernoulli(probs, generator=generator)

    # if output_dim == 1:
    #     y0 = y[0, :, 0]
    #     num_zeros = (y0 == 0).sum().item()
    #     num_ones = (y0 == 1).sum().item()
    #     total = y0.numel()
    #     print(probs[0,0:5:,0])
    #     print(f"Prevalence in y[0,:,0]: 0 -> {num_zeros/total:.2%}, 1 -> {num_ones/total:.2%}")

    # Per-sample (e.g., b=0)
    # if label:
    #     K = probs.size(-1)
    #     y0 = y[0,:,:]
    #     y_idx = y0.argmax(dim=-1) 
    #     counts_b = torch.bincount(y_idx.reshape(-1), minlength=K)
    #     prev = counts_b.float() / y_idx.numel()
    #     print("Prevalence in y[0]:", ", ".join([f"class {k}: {p:.2%}" for k,p in enumerate(prev)]))

    if return_groups:
        group_dict = {
            "groups": torch.stack(groups),
            "feature_indices": batch_feature_indices
        }

        return y, group_dict
    
    if return_p:
        return y, probs
    
    return y

class RealDataSampler(object):
    def __init__(self, train_dataset, test_dataset, seed=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dim_x = train_dataset.feature_dim
        self.dim_y = train_dataset.dim_y
        self.free_indices = train_dataset.free_indices
        self.feature_groups = train_dataset.feature_groups
        self.num_available_features = train_dataset.num_available_features

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    @staticmethod
    def _sample_indices(dataset: TabularDataset, size: int, batch_size: int, seed=None) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            return torch.stack([
                torch.randperm(len(dataset), generator=generator)[:size]
            for _ in range(batch_size)
        ])
        else:
            return torch.stack([
                torch.randperm(len(dataset))[:size]
            for _ in range(batch_size)
        ])
    
    @staticmethod
    def _normalize(x):
        # Normalize features
        mean = x.mean(dim=1, keepdim=True)     # [B, 1, Dx]
        std = x.std(dim=1, keepdim=True) + 1e-8  # [B, 1, Dx] (avoid div by zero)
        x = (x - mean) / std
        return x, mean, std
    
    def _generate_missingness(self, x, num_observed=None, seed=None, missing_rate=None):
        """Generate random missingness mask for each sample in batch.
        """
        batch_size, seq_len, num_features = x.shape
        
        if num_observed is None:
            # Randomly mask features
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
                unif = torch.rand(batch_size, seq_len, num_features, generator=generator)
                if missing_rate is not None:
                    ref = torch.full((batch_size, seq_len, 1), missing_rate)
                else:
                    ref = torch.rand(batch_size, seq_len, 1, generator=generator)
            else:
                unif = torch.rand(batch_size, seq_len, num_features)
                if missing_rate is not None:
                    ref = torch.full((batch_size, seq_len, 1), missing_rate)
                else:   
                    ref = torch.rand(batch_size, seq_len, 1)
                    
            mask = (unif > ref).float().to(x.device)
        else:
            if num_observed == 0:
                mask = torch.zeros_like(x)
            elif num_observed == num_features:
                mask = torch.ones_like(x)
            else:
                # Handle free indices if they exist
                if self.free_indices is not None:
                    
                    # Only randomly mask non-free indices
                    non_free_indices = [i for i in range(num_features) if i not in self.free_indices]
                    non_free_indices = torch.tensor(non_free_indices, device=x.device)
                    observed_features = torch.empty(batch_size, seq_len, num_observed, dtype=torch.long, device=x.device)
                    for b in range(batch_size):
                        for t in range(seq_len):
                            perm = torch.randperm(len(non_free_indices), device=x.device)[:num_observed]
                            observed_features[b, t] = non_free_indices[perm]
                    # Update mask with randomly selected features
                    mask = torch.zeros_like(x)
                    mask.scatter_(2, observed_features, 1.0)
                else:
                    rand_scores = torch.rand(batch_size, seq_len, num_features, device=x.device)
                    _, observed_features = torch.topk(rand_scores, num_observed, dim=2)
                    mask = torch.zeros_like(x)
                    mask.scatter_(2, observed_features, 1.0)

        if self.free_indices is not None:
            mask[:, :, self.free_indices] = 1.0
                
        return mask

    def sample(self, batch_size=16, num_ctx=None, num_tar=None, max_num_points=1000, device='cpu', num_observed=None, seed=None, return_groups=False):

        batch = {}
        num_ctx = num_ctx or 50
        num_tar = max_num_points - num_ctx
        num_points = num_ctx + num_tar  # N = Nc + Nt

        # Sample random sequence from dataset
        all_indices = self._sample_indices(self.train_dataset, num_points, batch_size, seed=seed)
        x = self.train_dataset.data[all_indices]  # [B,N,Dx]

        # Normalize data
        x, _, _ = self._normalize(x)
        
        # Sample retrospective MAR missingness mechanisms
        r = 1 - sample_random_function(x, output_dim=self.dim_x, max_prob=0.5, informative_features=self.free_indices, seed=seed)
        r[:, :, self.free_indices] = 1.0
        
        if self.train_dataset.mask is not None:
            r *= self.train_dataset.mask[all_indices]

        # Generate target state missingness mask: 1 = observed, 0 = missing
        maskt = self._generate_missingness(x[:, num_ctx:], num_observed=num_observed)
        mask = torch.ones_like(x[:, :, :self.dim_x])
        mask[:,num_ctx:,:] *= maskt 
        mask *= r

        # Ensure free indices are always observed in mask
        if self.free_indices is not None:
            mask[:, :, self.free_indices] = 1.0
        
        batch['x'] = x.to(device)
        batch['xc'] = batch['x'][:, :num_ctx]  # [B,Nc,2*Dx]
        batch['xt'] = batch['x'][:, num_ctx:]  # [B,Nt,2*Dx]
        
        if return_groups:
            batch['y'], groups = sample_random_function(x, output_dim=self.dim_y, seed=seed, return_groups=True, label=True)
            batch['groups'] = groups["groups"][:, num_ctx:]
            batch['group_feature_indices'] = groups["feature_indices"]
        else:
            batch['y'] = sample_random_function(x, output_dim=self.dim_y, seed=seed, label=True)
            
        batch['y'] = batch['y'].to(device)
        batch['yc'] = batch['y'][:, :num_ctx]  # [B,Nc,1]
        batch['yt'] = batch['y'][:, num_ctx:]  # [B,Nt,1]

        batch['mask'] = mask.to(device)
        batch['maskc'] = batch['mask'][:, :num_ctx]
        batch['maskt'] = batch['mask'][:, num_ctx:]

        batch['r'] = r.to(device)
        batch['rc'] = batch['r'][:, :num_ctx]
        batch['rt'] = batch['r'][:, num_ctx:]

        return batch
    
    def sample_test(self, 
                    batch_size=16, 
                    num_ctx=None, 
                    num_tar=None, 
                    max_num_points=1000, 
                    device='cpu', 
                    num_observed=None, 
                    unseen=False, 
                    real_y=False, 
                    subgroups=None,
                    max_p_missing=0.5,
                    min_informative=1,
                    max_informative=None,
                    return_groups=False,
                    seed=None):
        batch = {}

        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            num_ctx = num_ctx or torch.randint(low=50, high=max_num_points-50, size=[1], generator=g).item()  # Nc
            num_tar = num_tar or torch.randint(low=50, high=max_num_points-num_ctx, size=[1], generator=g).item()  # Nt
        else:
            num_ctx = num_ctx or torch.randint(low=50, high=max_num_points-50, size=[1]).item()  # Nc
            num_tar = num_tar or torch.randint(low=50, high=max_num_points-num_ctx, size=[1]).item()  # Nt

        num_points = num_ctx + num_tar  # N = Nc + Nt
        # Sample random context sequence from train dataset and target sequence from test dataset
        if unseen:
            all_indices = self._sample_indices(self.test_dataset, num_points, batch_size, seed=seed)
            x = self.test_dataset.data[all_indices]
            
            if real_y:
                y = self.test_dataset.label[all_indices]

        else:
            ctx_indices = self._sample_indices(self.train_dataset, num_ctx, batch_size, seed=seed)
            tar_indices = self._sample_indices(self.test_dataset, num_tar, batch_size, seed=seed)
            x = self.train_dataset.data[ctx_indices]  # [B,num_ctx,Dx]
            x_tar = self.test_dataset.data[tar_indices] # [B,num_tar,Dx]
            x = torch.cat([x, x_tar], dim=1)  # [B,N,Dx]

            if real_y:
                y = self.train_dataset.label[ctx_indices]
                y_tar = self.test_dataset.label[tar_indices]
                y = torch.cat([y, y_tar], dim=1)

        # Normalize data
        x_ctx, mean, std = self._normalize(x[:, :num_ctx])
        x_tar = (x[:, num_ctx:] - mean) / std
        x = torch.cat([x_ctx, x_tar], dim=1)
        
        # Sample missingness mechanisms
        r = 1 - sample_random_function(x, output_dim=self.dim_x, max_prob=max_p_missing, informative_features=self.free_indices, seed=seed)
        r[:, :, self.free_indices] = 1.0

        # Target fully observed for evaluation
        r[:, num_ctx:] = 1.0

        if self.train_dataset.mask is not None:
            if unseen:
                r *= self.test_dataset.mask[all_indices]
            else:
                rc = self.train_dataset.mask[ctx_indices]
                rt = self.test_dataset.mask[tar_indices]
                r *= torch.cat([rc, rt], dim=1)

        mask = torch.ones_like(x[:, :, :self.dim_x])
        maskt = self._generate_missingness(x[:, num_ctx:], num_observed)
        mask[:,num_ctx:,:] *= maskt 

        # Ensure free indices are always observed in mask
        if self.free_indices is not None:
            mask[:, :, self.free_indices] = 1.0

        batch['x'] = x.to(device)
        batch['xc'] = batch['x'][:, :num_ctx]  # [B,Nc,2*Dx]
        batch['xt'] = batch['x'][:, num_ctx:]  # [B,Nt,2*Dx]
        
        if real_y:
            if self.dim_y == 1:
                batch['y'] = y.unsqueeze(-1)
            else:
                batch['y'] = y
        else:
            if return_groups:
                labels, groups = sample_random_function(
                    x, 
                    output_dim=self.dim_y, 
                    subgroups=subgroups, 
                    seed=seed, 
                    min_feats=min_informative, 
                    max_feats=max_informative, 
                    return_groups=True,
                    label=True
                )
                batch['y'] = labels
                batch['groups'] = groups["groups"][:, num_ctx:]
                batch['group_feature_indices'] = groups["feature_indices"]
            else:
                batch['y'], batch['p'] = sample_random_function(
                    x, 
                    output_dim=self.dim_y, 
                    subgroups=subgroups, 
                    seed=seed, 
                    min_feats=min_informative, 
                    max_feats=max_informative,
                    label=True,
                    return_p=True
                    )
                
                batch['p'] = batch['p'].to(device)
                batch['pc'] = batch['p'][:, :num_ctx]  # [B,Nc,1]
                batch['pt'] = batch['p'][:, num_ctx:]  # [B,Nt,1]

        batch['y'] = batch['y'].to(device)
        batch['yc'] = batch['y'][:, :num_ctx]  # [B,Nc,1]
        batch['yt'] = batch['y'][:, num_ctx:]  # [B,Nt,1]

        batch['mask'] = mask.to(device)
        batch['maskc'] = batch['mask'][:, :num_ctx]
        batch['maskt'] = batch['mask'][:, num_ctx:]

        batch['r'] = r.to(device)
        batch['rc'] = batch['r'][:, :num_ctx]
        batch['rt'] = batch['r'][:, num_ctx:]

        return batch
    
    def sample_context(self, batch, num_ctx):
        new_ctx = batch['xc'][:, :num_ctx].clone()
        new_ctx_y = batch['yc'][:, :num_ctx].clone()
        new_ctx_mask = batch['maskc'][:, :num_ctx].clone()
        new_ctx_r = batch['rc'][:, :num_ctx].clone()

        x = torch.cat([new_ctx, batch['xt']], dim=1)
        y = torch.cat([new_ctx_y, batch['yt']], dim=1)
        r = torch.cat([new_ctx_r, batch['rt']], dim=1)
        mask = torch.cat([new_ctx_mask, batch['maskt']], dim=1)

        new_batch = {}
        new_batch['x'] = x
        new_batch['xc'] = new_ctx  # [B,Nc,2*Dx]
        new_batch['xt'] = batch['xt']

        new_batch['y'] = y
        new_batch['yc'] = new_ctx_y
        new_batch['yt'] = batch['yt']

        new_batch['mask'] = mask
        new_batch['maskc'] = new_ctx_mask
        new_batch['maskt'] = batch['maskt']

        new_batch['r'] = r
        new_batch['rc'] = new_ctx_r
        new_batch['rt'] = batch['rt']

        if 'p' in batch:
            new_ctx_p = batch['pc'][:, :num_ctx].clone()
            p = torch.cat([new_ctx_p, batch['pt']], dim=1)

            new_batch['p'] = p
            new_batch['pc'] = new_ctx_p
            new_batch['pt'] = batch['pt']

        return new_batch
    
    def mask_features(self, batch, num_observed=None, mask=None, seed=None, missing_rate=None):
        x = batch['xt']

        # Sample target state missingness mask if not provided
        if mask is None:
            mask = self._generate_missingness(x, num_observed=num_observed, seed=seed, missing_rate=missing_rate)

        new_batch = {}
        for k in ("x", "xc", "xt", "y", "yc", "yt", "r", "rc", "rt", "maskc", "p", "pc", "pt"):
            if k in batch:
                new_batch[k] = batch[k] 

        new_batch['mask'] = torch.cat((batch['maskc'], mask), dim=1)
        new_batch['maskt'] = mask

        return new_batch
    
    def initialize_mask(self, batch):
        maskt = torch.zeros_like(batch['maskt'])

        if self.free_indices is not None:
            maskt[:, :, self.free_indices] = 1.0
        
        return maskt
    
    def acquire_features(self, batch, mask, action=None, seed=None):
        batch_size, seq_len, dim_x = batch['maskt'].shape
        xt = batch['xt']
        rt = batch['rt'] if 'rt' in batch else None

        # If mask is not provided, initialize it to all zeros
        if mask is None:
            mask = torch.zeros_like(batch['maskt'])

            if self.free_indices is not None:
                mask[:, :, self.free_indices] = 1.0

        # If action is not provided, randomly select an action
        if action is None:
            # Create new mask
            new_mask = mask.clone()

            if rt is not None:
                available_mask = (mask == 0) & (rt == 1)  # shape: (B, T, F)
            else:
                available_mask = (mask == 0)

            B, T, F = available_mask.shape

            if seed is not None:
                g = torch.Generator(device=available_mask.device).manual_seed(seed)
                logits = torch.randn(B, T, F, generator=g, device=available_mask.device)
            else:
                logits = torch.randn(B, T, F, device=available_mask.device)

            masked_logits = logits.masked_fill(~available_mask, -1e9)
            idx = masked_logits.argmax(dim=-1)

            new_mask.scatter_(-1, idx.unsqueeze(-1), 1.0)

        else:
            new_mask = mask.clone()
            batch_idx = torch.arange(batch_size, device=action.device)[:, None]    # shape [B, 1]
            time_idx  = torch.arange(seq_len, device=action.device)[None, :]     # shape [1, T]
            valid = rt[batch_idx, time_idx, action] == 1
            new_mask[batch_idx, time_idx, action] = valid.float()

        new_batch = {}

        new_batch['mask'] = torch.cat([batch['maskc'], new_mask], dim=1)
        new_batch['maskc'] = batch['maskc']
        new_batch['maskt'] = new_mask

        for k in ("x", "xc", "xt", "y", "yc", "yt", "r", "rc", "rt", "p", "pc", "pt"):
            if k in batch:
                new_batch[k] = batch[k] 
    
        return new_batch, new_mask

