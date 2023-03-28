
from torch.utils.data import Dataset, TensorDataset, Subset


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


class LazyDataset(Dataset):
    """ Dataset which can lazily load objects on request """
    def __init__(self):
        super().__init__()
        self.items = {}

    def __getitem__(self, item):
        """ Load or fetch the value corresponding to the item """
        if item in self.items:
            return self.items[item]

        value = self.load(item)
        self.items[item] = value

        return value

    def __len__(self):
        raise NotImplementedError

    def load(self, item):
        """ Load and return the value corresponding to the item """
        raise NotImplementedError

    def clear(self):
        """ Clear items to free memory """
        self.items.clear()


import pathlib
import torch


class MeshDataset(LazyDataset):
    ALL_SHAPES = [
        "bathtub",
        "bed",
        "chair",
        "desk",
        "dresser",
        "monitor",
        "night_stand",
        "sofa",
        "table",
        "toilet",
    ]

    def __init__(self, root, shapes="all", split="train", samples: int = 2048):
        super().__init__()

        if shapes == "all":
            shapes = self.ALL_SHAPES
        for shape in shapes:
            if shape not in self.ALL_SHAPES:
                raise ValueError(f"Unknown shape: {shape}")

        self.root = pathlib.Path(root)
        self.shapes = shapes
        self.split = pathlib.Path(split)
        self.samples = samples

        self.shape_counts = {}

        for shape in self.shapes:
            path = self.root / shape / self.split
            files = list(path.glob("*.pt"))

            self.shape_counts[shape] = len(files)

    def __getitem__(self, item):
        points = super().__getitem__(item)

        indices = torch.randperm(len(points))[:self.samples]

        return points[indices]

    def __len__(self):
        return sum(self.shape_counts.values())

    def load(self, item):
        shape, item = self.shape_item(item)
        shape = pathlib.Path(shape)

        path = self.root / shape / self.split / f"{shape}_{item + 1:04d}.pt"

        return torch.load(path)

    def shape_item(self, item):
        """ Find the shape and corresponding index for a raw index """
        length = len(self)
        if not (-length <= item < length):
            raise IndexError(f"Index {item} is out of range for {self.__class__.__name__} with length {length}.")

        if item < 0:
            item += length

        for shape, count in self.shape_counts.items():
            if item < count:
                return shape, item

            item -= count
