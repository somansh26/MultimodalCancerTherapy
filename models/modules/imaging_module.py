import torch.nn as nn

class ImagingModule(nn.Module):
    def __init__(self):
        super(ImagingModule, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))
