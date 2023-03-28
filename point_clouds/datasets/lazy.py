
from torch.utils.data import Dataset


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
