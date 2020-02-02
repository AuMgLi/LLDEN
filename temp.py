import torch

from models import FeedForward, AlexNet

# model = FeedForward(num_classes=10)
"""
name: classifier.0.weight , param size: torch.Size([312, 784])
name: classifier.0.bias , param size: torch.Size([312])
name: classifier.2.weight , param size: torch.Size([128, 312])
name: classifier.2.bias , param size: torch.Size([128])
name: classifier.4.weight , param size: torch.Size([10, 128])
name: classifier.4.bias , param size: torch.Size([10])
"""
model = AlexNet(num_classes=10)
"""
name: features.0.weight , param size: torch.Size([64, 3, 11, 11])
name: features.0.bias , param size: torch.Size([64])
name: features.3.weight , param size: torch.Size([192, 64, 5, 5])
name: features.3.bias , param size: torch.Size([192])
name: features.6.weight , param size: torch.Size([384, 192, 3, 3])
name: features.6.bias , param size: torch.Size([384])
name: features.8.weight , param size: torch.Size([256, 384, 3, 3])
name: features.8.bias , param size: torch.Size([256])
name: features.10.weight , param size: torch.Size([256, 256, 3, 3])
name: features.10.bias , param size: torch.Size([256])
name: classifier.1.weight , param size: torch.Size([4096, 9216])
name: classifier.1.bias , param size: torch.Size([4096])
name: classifier.4.weight , param size: torch.Size([4096, 4096])
name: classifier.4.bias , param size: torch.Size([4096])
name: classifier.6.weight , param size: torch.Size([10, 4096])
name: classifier.6.bias , param size: torch.Size([10])
"""
# print(list(model.named_parameters()))
# for name, param in model.named_parameters():
#     print('name:', name, ', param size:', param.data)

w1 = torch.randn(192, 64).normal_(0, 0.005)
print(w1)
w2 = torch.zeros(6)
print(w2)
# print(w1.shape)
# w1 = torch.cat(
#     (w1, torch.randn(6)),
#     dim=0,
# )
# print(w1.shape)
# w1 = torch.cat(
#     (w1, torch.randn(192+6, 6, 5, 5)),
#     dim=1,
# )
# print(w1.shape)
