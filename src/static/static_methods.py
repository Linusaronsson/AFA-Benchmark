import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import RelaxedOneHotCategorical
import numpy as np
from static.utils import restore_parameters
from static.models import get_network, BaseModel
from copy import deepcopy
from pathlib import Path
from common.custom_types import AFAMethod, AFAPredictFn, MaskedFeatures, FeatureMask, Features, Label, AFASelection


class ConcreteMask(nn.Module):
    '''
    For differentiable global feature selection.
    '''

    def __init__(self, num_features, num_select, group_matrix=None, append=False, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, num_features, dtype=torch.float32))
        self.append = append
        self.gamma = gamma
        if group_matrix is None:
            self.group_matrix = None
        else:
            self.register_buffer('group_matrix', group_matrix.float())

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        if self.group_matrix is not None:
            out = x * (m @ self.group_matrix)
        else:
            out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    

class DifferentiableSelector(nn.Module):
    '''Differentiable global feature selection.'''

    def __init__(self, model, selector_layer):
        super().__init__()
        self.model = model
        self.selector_layer = selector_layer
    
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-5,
            early_stopping_epochs=None,
            start_temp=10.0,
            end_temp=0.01,
            temp_steps=10,
            verbose=True):
        '''
        Train model to perform global feature selection.
        '''
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        
        # More setup.
        model = self.model
        selector_layer = self.selector_layer
        device = next(model.parameters()).device
        val_loss_fn.to(device)

        # For tracking best models with zero temperature.
        best_val = None
        best_zerotemp_model = None
        best_zerotemp_selector = None
        
        # Train separately with each temperature.
        total_epochs = 0
        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                print(f'Starting training with temp = {temp:.4f}\n')

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(list(model.parameters()) + list(selector_layer.parameters()), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=val_loss_mode, factor=factor, patience=patience,
                min_lr=min_lr)
            
            # For tracking best models and early stopping.
            best_model = deepcopy(model)
            best_selector = deepcopy(selector_layer)
            num_bad_epochs = 0
            epoch = 0
            for epoch in range(nepochs):
                # Switch model to training mode.
                model.train()

                for x, y in train_loader:
                    # Move to device.
                    x = x.to(device)
                    y = y.to(device)
                    
                    # Select features and make prediction.
                    x_masked = selector_layer(x, temp)
                    pred = model(x_masked)

                    # Calculate loss.
                    loss = loss_fn(pred, y)

                    # Take gradient step.
                    loss.backward()
                    opt.step()
                    model.zero_grad()
                    selector_layer.zero_grad()
                    
                # Reinitialize logits as necessary.
                logits = selector_layer.logits
                argmax = logits.argmax(dim=1).cpu().data.numpy()
                selected = []
                for i, ind in enumerate(argmax):
                    if ind in selected:
                        logits.data[i] = 0
                    else:
                        selected.append(ind)

                # Calculate validation loss.
                model.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []

                    for x, y in val_loader:
                        # Move to device.
                        x = x.to(device)

                        # Evaluate model with soft sample.
                        x_masked = selector_layer(x, temp)
                        pred = model(x_masked)
                        
                        # Evaluate model with hard sample.
                        x_masked = selector_layer(x, 1e-6)
                        hard_pred = model(x_masked)
                        
                        # Append to lists.
                        pred_list.append(pred.cpu())
                        hard_pred_list.append(hard_pred.cpu())
                        label_list.append(y.cpu())
                        
                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = val_loss_fn(pred, y)
                    val_hard_loss = val_loss_fn(hard_pred, y)
                
                # Print progress.
                if verbose:
                    print(f'{"-"*8}Epoch {epoch+1} ({epoch + 1 + total_epochs} total){"-"*8}')
                    print(f'Val loss = {val_loss:.4f}, Zero-temp loss = {val_hard_loss:.4f}\n')
                    
                # Update scheduler.
                scheduler.step(val_loss)

                # See if best model.
                if val_loss == scheduler.best:
                    best_model = deepcopy(model)
                    best_selector = deepcopy(selector_layer)
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1
                    
                # Check if best model with zero temperature.
                if ((best_val is None)
                    or (val_loss_mode == 'min' and val_hard_loss < best_val)
                    or (val_loss_mode == 'max' and val_hard_loss > best_val)):
                    best_val = val_hard_loss
                    best_zerotemp_model = deepcopy(model)
                    best_zerotemp_selector = deepcopy(selector_layer)
                    
                # Early stopping.
                if num_bad_epochs > early_stopping_epochs:
                    break

            # Update total epoch count.
            if verbose:
                print(f'Stopping temp = {temp:.4f} at epoch {epoch+1}\n')
            total_epochs += (epoch + 1)
            
            # Copy parameters from best model.
            restore_parameters(model, best_model)
            restore_parameters(selector_layer, best_selector)
            
        # Copy parameters from best model with zero temperature.
        restore_parameters(model, best_zerotemp_model)
        restore_parameters(selector_layer, best_zerotemp_selector)


class StaticBaseMethod(AFAMethod):
    def __init__(
        self, selected_history: dict[int, list[int]],
        predictors: dict[int, nn.Module],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.selected_history = selected_history
        self.predictors = {b: m.to(device) for b, m in predictors.items()}
        self._device = device

    def predict(self, masked_features: MaskedFeatures, feature_mask: FeatureMask,
            features=None, label=None) -> Label:
        counts = feature_mask.sum(dim=1)
        if not (counts == counts[0]).all():
            raise RuntimeError("mixed budgets in batch")
        b = int(counts[0].item())
        cols = self.selected_history[b]
        x_sel = masked_features[:, cols].to(self._device)
        logits = self.predictors[b](x_sel)
        return logits.softmax(dim=-1)
    
    def select(self, masked_features: MaskedFeatures, feature_mask: FeatureMask,
            features=None, label=None) -> AFASelection:
        counts = feature_mask.sum(dim=1)
        if not (counts == counts[0]).all():
            raise RuntimeError("mixed budgets in batch")
        b = int(counts[0].item())
        mask0 = feature_mask[0]
        for idx in self.selected_history[b+1]:
            if mask0[idx] == 0:
                choice = idx
                return torch.full(
                    (masked_features.size(0),),
                    fill_value=choice,
                    dtype=torch.long,
                    device=self._device
                )
        
        return torch.zeros((masked_features.size(0),), dtype=torch.long, device=self._device)
    
    def save(self, path: Path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "selected_history": self.selected_history
        }, path / "selected.pt")
        for b, mdl in self.predictors.items():
            torch.save(mdl, path / f"predictor_b{b}.pt")

    @classmethod
    def load(cls, path: Path, device="cpu"):
        data = torch.load(path / "selected.pt", weights_only=False, map_location="cpu")
        hist = data["selected_history"]
        preds = {}
        for b in hist.keys():
            model = torch.load(path / f"predictor_b{b}.pt", weights_only=False, map_location=device)
            preds[b] = model.to(device)
        
        return cls(hist, preds, device).to(device)

    def to(self, device):
        self._device = device
        for b in self.predictors:
            self.predictors[b] = self.predictors[b].to(device)
        return self
    
    @property
    def device(self):
        return self._device
