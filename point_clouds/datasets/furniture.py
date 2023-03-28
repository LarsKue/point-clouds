import pathlib
import torch
from torchvision.datasets.utils import download_and_extract_archive

import trimesh
import numpy as np
from tqdm import tqdm

from .lazy import LazyDataset


class FurnitureDataset(LazyDataset):
    SPLITS = [
        "train",
        "test",
    ]
    SHAPES = [
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

    def __init__(self, root, shapes="all", split="train", sub_samples: int = 2048, samples: int = 32768, download=True):
        super().__init__()

        if shapes == "all":
            shapes = self.SHAPES
        if isinstance(shapes, str):
            shapes = [shapes]
        for shape in shapes:
            if shape not in self.SHAPES:
                raise ValueError(f"Unknown shape: {shape}")

        self.root = pathlib.Path(root)
        self.shapes = shapes
        self.split = pathlib.Path(split)
        self.samples = samples
        self.sub_samples = sub_samples

        if download:
            if self.root.exists():
                print("Root already exists. Skipping download...")
            else:
                self.download()
                self.resample(self.samples)

        self.shape_counts = {}

        for shape in self.shapes:
            path = self.root / "processed" / shape / self.split
            files = list(path.glob("*.pt"))

            self.shape_counts[shape] = len(files)

    def __getitem__(self, item):
        points = super().__getitem__(item)

        indices = torch.randperm(len(points))[:self.sub_samples]

        return points[indices]

    def __len__(self):
        return sum(self.shape_counts.values())

    def download(self):
        url = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        filename = "ModelNet10.zip"
        download_and_extract_archive(url, download_root=str(self.root), extract_root=str(self.root), filename=filename)

    def resample(self, samples: int):
        data_root = self.root / "ModelNet10"
        processed = self.root / "processed"

        print("Supersampling shapes...")
        for split in tqdm(self.SPLITS, desc="Splits"):
            for shape in tqdm(self.SHAPES, desc="Shapes"):
                path = data_root / shape / split
                new_path = processed / shape / split
                for i, file in enumerate(path.glob("*.off")):
                    mesh = trimesh.load(file)
                    points = mesh.sample(samples).astype(np.float32)
                    points = torch.from_numpy(points)

                    filepath = new_path / file.with_suffix(".pt").name
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(points, new_path / file.with_suffix(".pt").name)

        self.samples = samples

    def load(self, item):
        shape, item = self.shape_item(item)
        shape = pathlib.Path(shape)

        path = self.root / "processed" / shape / self.split / f"{shape}_{item + 1:04d}.pt"

        try:
            data = torch.load(path)
        except FileNotFoundError as error:
            raise FileNotFoundError("Could not find processed data tensor. "
                                    "Did you forget to download/resample?").with_traceback(error.__traceback__)

        return data

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
