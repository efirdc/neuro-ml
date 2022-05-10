from typing import Sequence, Optional

from torch import nn


class DenseNetwork(nn.Module):
    def __init__(
            self,
            layer_sizes: Sequence[int],
            dropout_p: Optional[float] = None,
    ):
        super().__init__()
        self.layers = [nn.Linear(in_features=layer_sizes[0], out_features=layer_sizes[1])]
        for in_size, out_size in zip(layer_sizes[1:-1], layer_sizes[2:]):
            self.layers += [
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(in_size, out_size)
            ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x
