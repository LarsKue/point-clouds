
import torch
import pathlib
import trimesh
import numpy as np

n = 32768
assert np.log2(n) % 1 == 0


data_root = "data/ModelNet10"
splits = ["train", "test"]
shapes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]


root = pathlib.Path(data_root)
processed = pathlib.Path("data/processed")

for split in splits:
    print(split)
    for shape in shapes:
        print(shape)
        path = root / shape / split
        new_path = processed / shape / split
        for i, file in enumerate(path.glob("*.off")):
            mesh = trimesh.load(file)
            points = mesh.sample(n).astype(np.float32)
            points = torch.from_numpy(points)

            filepath = new_path / file.with_suffix(".pt").name
            filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(points, new_path / file.with_suffix(".pt").name)

