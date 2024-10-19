import torch

from nora.config import LazyCall as L
from nora.solver.optimizer import get_default_optimizer_params

SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(weight_decay_norm=0),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)

AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
