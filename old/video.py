
import pathlib as pl
import re

import imageio.v2 as iio

base_path = pl.Path("plots/quiver")

pattern = "t=(.*).png"

files = {}

for file in base_path.glob("*"):
    match = re.match(pattern, file.name)
    if match:
        files[match.group(1)] = file.name

files = {key: value for key, value in sorted(files.items(), key=lambda pair: float(pair[0]))}

writer = iio.get_writer("video.avi", fps=60)

for time, filename in files.items():
    print(f"\rt={float(time):.02f}", end="")
    img = iio.imread(base_path / filename)
    writer.append_data(img)
