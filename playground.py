
from data import MeshDataset
from torch.utils.data import DataLoader
import torch

data_root = "data/ModelNet10"

# check dtype
ds = MeshDataset(root=data_root)

print(ds[0].shape)
print(ds[1].shape)
print(ds[-1].shape)
print(ds[3000].shape)
print(ds[-3000].shape)
print(len(ds))
print(ds[len(ds) - 1].shape)
print(ds[-len(ds) + 1].shape)

print(ds[-len(ds)] is ds[0])
print(torch.allclose(ds[-len(ds)], ds[0]))
# print(ds[100000000])
# print(ds[-1000000000])
