import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

class AlexNetCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCifar, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.classifier(x)
        return x

class AlexNetDownCifar(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDownCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, x):
        x = self.model(x)
        return x

local_model = True
if local_model is True:
    model = AlexNetDownCifar()

else:
    model = AlexNetCifar(num_classes=10)

dummy_input = torch.randn(1, 3, 32, 32)

# Calculate FLOPs using fvcore
flops = FlopCountAnalysis(model, dummy_input)
print("FLOPs:", flops.total())

# Calculate Parameters using fvcore
params_table = parameter_count_table(model)
print(params_table)

# Calculate total parameters
total_params = parameter_count(model)[""]
print("Total parameters:", total_params)

# Each parameter in PyTorch by default uses 32 bits
total_bits = total_params * 32
print("Total bits:", total_bits)

# Calculate total bytes assuming float32 storage
total_bytes = total_params * 4  # each float32 parameter takes 4 bytes

# Convert bytes to megabytes
total_megabytes = total_bytes / (1024 ** 2)
print(f"Total size in MB: {total_megabytes:.2f} MB")

