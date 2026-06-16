from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
