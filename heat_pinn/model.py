from __future__ import annotations

"""Ορισμός του νευρωνικού δικτύου του PINN"""

import torch
import torch.nn as nn


class HeatPINN(nn.Module):
    """Δίκτυο για την 2D εξίσωση θερμότητας"""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] | list[int],
        *,
        t_min: float = 0.0,
        t_max: float = 1.0,
        x_min: float = -1.0,
        x_max: float = 1.0,
        y_min: float = -1.0,
        y_max: float = 1.0,
    ) -> None:
        super().__init__()
        # Χρησιμοποιώ tanh σε όλα τα κρυφά layers
        self.activation = nn.Tanh()
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.layers = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        )

    @staticmethod
    def _normalize(
        values: torch.Tensor,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        """Φέρνει κάθε είσοδο στο [-1, 1]"""

        return 2.0 * (values - lower) / (upper - lower) - 1.0

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ενώνω πρώτα τις κανονικοποιημένες εισόδους
        z = torch.cat(
            [
                self._normalize(t, self.t_min, self.t_max),
                self._normalize(x, self.x_min, self.x_max),
                self._normalize(y, self.y_min, self.y_max),
            ],
            dim=1,
        )

        # Περνάω από όλα τα κρυφά layers
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))

        # Το τελευταίο layer δίνει την πρόβλεψη u(t, x, y)
        return self.layers[-1](z)
