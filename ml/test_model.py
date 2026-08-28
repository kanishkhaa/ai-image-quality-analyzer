import torch
from model import IQAModel


# Create model
model = IQAModel()

# Create dummy batch
x = torch.randn(16, 3, 224, 224)

# Forward pass
output = model(x)

print("===== MODEL TEST =====")
print("Input shape :", x.shape)
print("Output shape:", output.shape)
print("Output:")
print(output[:5])