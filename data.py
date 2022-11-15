
from torch.utils.data import Dataset, TensorDataset


class SingleTensorDataset(TensorDataset):
    def __getitem__(self, item):
        return super().__getitem__(item)[0]


class PairedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, item):
        return tuple(d[item] for d in self.datasets)


class ProductSet(Dataset):
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1) * len(self.d2)

    def __getitem__(self, item):
        i, j = item % len(self.d1), item // len(self.d1)
        return self.d1[i], self.d2[j]
