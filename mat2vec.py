import torch
a=torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape)
b=a.view(-1)
print(b.shape)
