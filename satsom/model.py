from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class SatSOMParameters:
    grid_shape: tuple[int, ...]  # e.g., (m, n, ...)
    input_dim: int  # dimensionality of input features
    output_dim: int  # dimensionality of output labels (e.g., number of classes)

    initial_lr: float  # initial learning rate for neurons
    initial_sigma: float  # initial neighborhood radius (sigma)

    Lr: float  # decay rate for learning rate
    Lr_bias: float  # bias multiplier for neighborhood in training
    Lr_sigma: float  # decay rate for sigma

    q: float = 0.005  # disable neurons beyond this quantile threshold
    p: float = 10.0  # higher p penalizes larger distances


class SatSOM(nn.Module):
    def __init__(self, params: SatSOMParameters):
        super().__init__()
        self.params = params
        self.num_neurons = int(np.prod(self.params.grid_shape))

        # Prototype weight vectors: (num_neurons, input_dim)
        self.weights = nn.Parameter(
            torch.randn(self.num_neurons, params.input_dim), requires_grad=False
        )

        # Per-neuron learning rates and sigmas
        self.learning_rates = nn.Parameter(
            torch.full((self.num_neurons,), params.initial_lr), requires_grad=False
        )
        self.sigmas = nn.Parameter(
            torch.full((self.num_neurons,), params.initial_sigma), requires_grad=False
        )

        # Spatial locations on the D-dimensional grid: (num_neurons, D)
        self.locations = nn.Parameter(
            self._create_locations(self.params.grid_shape), requires_grad=False
        )

        # Label prototypes for each neuron: (num_neurons, output_dim)
        self.labels = nn.Parameter(
            torch.zeros(self.num_neurons, params.output_dim), requires_grad=False
        )

    def _create_locations(self, shape: tuple[int, ...]) -> torch.Tensor:
        ranges = [torch.arange(s) for s in shape]
        coords = torch.cartesian_prod(*ranges).float()  # (num_neurons, D)
        return coords

    def forward(
        self, x: torch.Tensor, q: Optional[float] = None, p: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape[1] == self.params.input_dim

        q = q or self.params.q
        p = p or self.params.p

        # Compute distances (batch, num_neurons)
        dists = self.dists(x)
        dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-6)

        # Mask out under-trained neurons by saturation
        saturation = (
            self.params.initial_lr - self.learning_rates
        ) / self.params.initial_lr
        mask = saturation.repeat(batch_size, 1) < 1e-4  # under-saturated
        dists[mask] = 1.0

        # Disable neurons beyond quantile threshold
        thresh = dists[~mask].quantile(q) if mask.any() else float("inf")
        dists[dists > thresh] = 1.0

        # Convert to proximity
        proximity = 1 - dists.view(batch_size, self.num_neurons, 1)
        proximity = proximity**p

        # Broadcast neuron labels across batch: (batch, num_neurons, output_dim)
        labels_batch = self.labels.unsqueeze(0).repeat(batch_size, 1, 1)

        # Weighted sum
        scaled_labels = labels_batch * proximity
        out = scaled_labels.mean(dim=1)  # (batch, output_dim)

        return out

    def dists(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pairwise Euclidean distances between x and each neuron weight.
        Returns shape (batch, num_neurons)
        """
        x_flat = x.view(-1, self.params.input_dim)
        return torch.cdist(x_flat, self.weights)

    def find_bmu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Best matching unit index for each vector in x.
        """
        return torch.argmin(self.dists(x), dim=1)

    def _neighborhood(self, bmu_idx: int) -> torch.Tensor:
        """
        Compute D-dimensional Gaussian neighborhood around BMU using mixed sigma product.
        """
        locs = self.locations  # (num_neurons, D)
        sigs = self.sigmas  # (num_neurons,)
        bmu_loc = locs[bmu_idx]  # (D,)
        d2 = ((locs - bmu_loc) ** 2).sum(dim=1)

        sig_prod = sigs[bmu_idx] * sigs
        nbh = torch.exp(-d2 / (2 * sig_prod.clamp_min(1e-8)))

        if self.training:
            sat = (
                self.params.initial_lr - self.learning_rates
            ) / self.params.initial_lr
            nbh *= 1 + self.params.Lr_bias * (1 - sat)

        return nbh

    def step(self, x: torch.Tensor, label: torch.Tensor):
        """
        Single update step: find BMU, update weights, labels, and decay rates.
        x: single sample tensor shape (input_dim,)
        label: one-hot tensor shape (output_dim,)
        """

        bmu_idx = self.find_bmu(x.unsqueeze(0)).item()
        nbh = self._neighborhood(bmu_idx).unsqueeze(1)  # (num_neurons, 1)

        # Update weights
        lr = self.learning_rates.unsqueeze(1)
        self.weights += lr * nbh * (x - self.weights)

        # Update labels
        probs = torch.softmax(self.labels, dim=1)
        target = label.repeat(self.num_neurons, 1)
        grad = probs - target

        self.labels += -lr * nbh * grad

        # Decay learning rates and sigmas
        self.learning_rates *= torch.exp(-self.params.Lr * nbh.squeeze(1))
        self.sigmas *= torch.exp(-self.params.Lr_sigma * nbh.squeeze(1))
