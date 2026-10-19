# PATCH(afabench): import path of the vendored copy.
from afabench.components.methods.discriminative.l2m.upstream.ltm import LTM

def get_model(config):
    model = LTM(
        dim_x=config['feature_dim'] * 2,
        dim_y=config['label_dim'],
        d_model=config['d_model'],
        emb_depth=config['emb_depth'],
        pred_depth=config['pred_depth'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        task=config['task'],
        bound_std=config['bound_std'],
    )
    return model