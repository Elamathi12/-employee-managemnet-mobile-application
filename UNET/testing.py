import torch
from architecture_2 import UNetVGG  


model = UNetVGG(out_channels=1, in_channels=1)
print(model)
x = torch.randn(1, 1, 200, 200)
y = model(x)

print("output shape ",y.shape)
