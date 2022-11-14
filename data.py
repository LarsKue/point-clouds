
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
