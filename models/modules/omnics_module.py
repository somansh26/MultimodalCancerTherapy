import torch.nn as nn

class OmicsModule(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(OmicsModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
