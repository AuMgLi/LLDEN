import torch

from models import FeedForward

model = FeedForward(num_classes=10)

for name, param in model.named_parameters():
    print('name:', name, ', param:', param)
