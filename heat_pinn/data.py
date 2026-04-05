from __future__ import annotations

"""Δημιουργία των σημείων εκπαίδευσης του PINN"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
import torch

from .config import DomainConfig, SamplingConfig
from .problem import exact_solution


@dataclass
class TrainingData:
    """Όλα τα tensors που χρειάζονται στην εκπαίδευση"""

    t_f: torch.Tensor
    x_f: torch.Tensor
    y_f: torch.Tensor
    t_u: torch.Tensor
    x_u: torch.Tensor
    y_u: torch.Tensor
    u_real: torch.Tensor


def _sample_lhs_box(
    num_points: int,
    lower_bounds: tuple[float, ...],
    upper_bounds: tuple[float, ...],
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Κάνει Latin Hypercube Sampling σε κουτί ορίων"""

    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unit_samples = sampler.random(n=num_points)
    return qmc.scale(unit_samples, lower_bounds, upper_bounds)


def generate_training_data(
    domain: DomainConfig,
    sampling: SamplingConfig,
    device: torch.device,
    seed: int | None = None,
) -> TrainingData:
    """Φτιάχνει collocation, initial και boundary σημεία"""

    lhs_seed = seed if seed is not None else int(np.random.randint(0, 2**32 - 1))

    # Εσωτερικά σημεία για το residual της PDE
    collocation = _sample_lhs_box(
        sampling.n_f,
        (domain.t_min, domain.x_min, domain.y_min),
        (domain.t_max, domain.x_max, domain.y_max),
        seed=lhs_seed,
    )
    collocation_tensor = torch.tensor(collocation, dtype=torch.float32, device=device)
    t_f = collocation_tensor[:, 0:1]
    x_f = collocation_tensor[:, 1:2]
    y_f = collocation_tensor[:, 2:3]

    t_f.requires_grad_(True)
    x_f.requires_grad_(True)
    y_f.requires_grad_(True)

    # Σημεία αρχικής συνθήκης στο t = 0
    t_i = domain.t_min * torch.ones(sampling.n_i, 1, device=device)
    x_i = (domain.x_max - domain.x_min) * torch.rand(sampling.n_i, 1, device=device) + domain.x_min
    y_i = (domain.y_max - domain.y_min) * torch.rand(sampling.n_i, 1, device=device) + domain.y_min
    u_i = exact_solution(t_i, x_i, y_i)

    # Σημεία στα τέσσερα σύνορα του χωρίου
    t_b1 = (domain.t_max - domain.t_min) * torch.rand(sampling.n_b, 1, device=device) + domain.t_min
    x_b1 = domain.x_min * torch.ones(sampling.n_b, 1, device=device)
    y_b1 = (domain.y_max - domain.y_min) * torch.rand(sampling.n_b, 1, device=device) + domain.y_min
    u_b1 = exact_solution(t_b1, x_b1, y_b1)

    t_b2 = (domain.t_max - domain.t_min) * torch.rand(sampling.n_b, 1, device=device) + domain.t_min
    x_b2 = domain.x_max * torch.ones(sampling.n_b, 1, device=device)
    y_b2 = (domain.y_max - domain.y_min) * torch.rand(sampling.n_b, 1, device=device) + domain.y_min
    u_b2 = exact_solution(t_b2, x_b2, y_b2)

    t_b3 = (domain.t_max - domain.t_min) * torch.rand(sampling.n_b, 1, device=device) + domain.t_min
    x_b3 = (domain.x_max - domain.x_min) * torch.rand(sampling.n_b, 1, device=device) + domain.x_min
    y_b3 = domain.y_min * torch.ones(sampling.n_b, 1, device=device)
    u_b3 = exact_solution(t_b3, x_b3, y_b3)

    t_b4 = (domain.t_max - domain.t_min) * torch.rand(sampling.n_b, 1, device=device) + domain.t_min
    x_b4 = (domain.x_max - domain.x_min) * torch.rand(sampling.n_b, 1, device=device) + domain.x_min
    y_b4 = domain.y_max * torch.ones(sampling.n_b, 1, device=device)
    u_b4 = exact_solution(t_b4, x_b4, y_b4)

    # Ενώνω όλα τα supervised σημεία σε ένα σύνολο
    t_u = torch.cat([t_i, t_b1, t_b2, t_b3, t_b4], dim=0)
    x_u = torch.cat([x_i, x_b1, x_b2, x_b3, x_b4], dim=0)
    y_u = torch.cat([y_i, y_b1, y_b2, y_b3, y_b4], dim=0)
    u_real = torch.cat([u_i, u_b1, u_b2, u_b3, u_b4], dim=0)

    return TrainingData(
        t_f=t_f,
        x_f=x_f,
        y_f=y_f,
        t_u=t_u,
        x_u=x_u,
        y_u=y_u,
        u_real=u_real,
    )
