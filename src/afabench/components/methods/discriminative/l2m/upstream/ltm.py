import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Bernoulli, Categorical

# PATCH(afabench): import path of the vendored copy.
from afabench.components.methods.discriminative.l2m.upstream.utils import MLP

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim), nn.GELU())
                for i in range(num_layers)
            ]
        )
        self.out = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        return self.ln(self.out(self.layers(x)) + residual)

class TNP(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        bound_std
    ):
        super(TNP, self).__init__()

        input_dim = dim_x + dim_y
        self.embedder = ResidualMLP(input_dim, d_model, dim_feedforward, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.bound_std = bound_std

    def construct_input(self, batch, autoreg=False):
        xc = batch['xc'] * batch['rc']
        xc_input = torch.cat([xc, batch['rc']], dim=-1)

        mt = batch['rt'] * batch['maskt']
        xt = batch['xt'] * mt
        xt_input = torch.cat([xt, mt], dim=-1)

        x_y_ctx = torch.cat((xc_input, batch['yc']), dim=-1)
        x_0_tar = torch.cat((xt_input, torch.zeros_like(batch['yt'])), dim=-1)

        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            xt_real = batch['xt'] * batch['rt']
            xt_real_input = torch.cat([xt_real, batch['rt']], dim=-1)
            x_y_tar = torch.cat((xt_real_input, batch['yt']), dim=-1)

            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask(self, batch, autoreg=False):
        num_ctx = batch['xc'].shape[1]
        num_tar = batch['xt'].shape[1]
        num_all = num_ctx + num_tar
        if not autoreg:
            mask = torch.zeros(num_all, num_all, device=batch['xc'].device).fill_(float('-inf'))  # PATCH(afabench): device of the batch, not 'cuda'
            mask[:, :num_ctx] = 0.0
            mask[num_ctx:, num_ctx:].fill_diagonal_(0)
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar), device=batch['xc'].device).fill_(float('-inf'))  # PATCH(afabench): device of the batch, not 'cuda'
            mask[:, :num_ctx] = 0.0 # all points attend to context points
            mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
            mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points
            mask[num_ctx:, num_ctx:].fill_diagonal_(0)

        return mask, num_tar

    def encode(self, batch, autoreg=False):
        mask, num_tar = self.create_mask(batch, autoreg)
        inp = self.construct_input(batch, autoreg)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask, is_causal=False)
        return out[:, -num_tar:]
    

class LTM(TNP):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        pred_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        task,
        bound_std=False
    ):
        super(LTM, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            bound_std
        )
        
        self.task = task
        self.dim_y = dim_y

        if task == 'regression':
            self.predictor = MLP(d_model, dim_y*2, dim_feedforward, pred_depth)
        elif task == 'classification':
            self.predictor = MLP(d_model, dim_y, dim_feedforward, pred_depth)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.selector = MLP(d_model, dim_x // 2, dim_feedforward, pred_depth, policy_head=True)

    def forward(self, batch, reduce_ll=True, autoreg=True):
        z_target = self.encode(batch, autoreg=autoreg)
        out = self.predictor(z_target)
        
        outs = {}
        if self.task == 'regression':
            mean, std = torch.chunk(out, 2, dim=-1)
            if self.bound_std:
                std = 0.05 + 0.95 * F.softplus(std)
            else:
                std = torch.exp(std)

            pred_tar = Normal(mean, std)

            if reduce_ll:
                outs['loss'] = -pred_tar.log_prob(batch['yt']).sum(-1).mean()
            else:
                outs['loss'] = -pred_tar.log_prob(batch['yt']).sum(-1).unsqueeze(-1)
        else:  # classification
            logits = out
            if self.dim_y == 1:
                if reduce_ll:
                    outs['loss'] = F.binary_cross_entropy_with_logits(logits, batch['yt'])
                else:
                    outs['loss'] = F.binary_cross_entropy_with_logits(logits, batch['yt'], reduction='none')
            else:
                targets = batch['yt'].argmax(dim=-1)  # [B, T]
                dist = Categorical(logits=logits)  # handles softmax internally

                if reduce_ll:
                    outs['loss'] = -dist.log_prob(targets).mean()
                else:
                    outs['loss'] = -dist.log_prob(targets)  # [B, T]
                    outs['loss'] = outs['loss'].unsqueeze(-1)

        return outs

    def predict(self, batch):
        new_batch = {}
        new_batch['xc'] = batch['xc']
        new_batch['yc'] = batch['yc']
        new_batch['xt'] = batch['xt']
        new_batch['yt'] = torch.zeros_like(batch['yt'])
        new_batch['rc'] = batch['rc']
        new_batch['rt'] = batch['rt']
        new_batch['maskt'] = batch['maskt']

        z_target = self.encode(new_batch, autoreg=False)
        out = self.predictor(z_target)
        
        if self.task == 'regression':
            mean, std = torch.chunk(out, 2, dim=-1)
            if self.bound_std:
                std = 0.05 + 0.95 * F.softplus(std)
            else:
                std = torch.exp(std)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(batch['yt'])
            return dist, log_probs
        else:  # classification
            if self.dim_y == 1:
                probs = torch.sigmoid(out)
                dist = Bernoulli(probs)
                log_probs = dist.log_prob(batch['yt'])
                return dist, log_probs
            else:
                dist = Categorical(logits=out)  
                log_probs = dist.log_prob(batch['yt'].argmax(dim=-1))
                return dist, log_probs

    def select_action(self, batch, autoreg=False):
        z = self.encode(batch, autoreg=autoreg)
        out = self.selector(z)
        return out