import torch.nn as nn

class MultiLayerNN(nn.Module):
    def __init__(self, latent_size=128, num_layers=3):
        super().__init__()
        self.dim_reduction = nn.Linear(28*28, latent_size)
        layers = []
        for i in range(num_layers):
            layers.extend([nn.Linear(latent_size, latent_size), nn.ReLU()])
        self.hidden_layers = nn.ModuleList(
            layers,
        )
        self.output = nn.Linear(latent_size, 10)

    def forward(self, x):
        x = self.dim_reduction(x)
        x = nn.ReLU()(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output(x)