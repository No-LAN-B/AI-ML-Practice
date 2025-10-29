import numpy as np

header = [
    "ply",
    "format binary_little_endian 1.0",
    "element vertex 100",
    "property float x",
    "property float y",
    "property float z",
    "property float scale_0",
    "property float scale_1",
    "property float scale_2",
    "property float rot_0",
    "property float rot_1",
    "property float rot_2",
    "property float rot_3",
    "property float f_dc_0",
    "property float f_dc_1",
    "property float f_dc_2",
    "property float opacity"
]
header += [f"property float sh_{i}" for i in range(45)]
header.append("end_header")

lines = ["\n".join(header)]
np.random.seed(42)
for _ in range(100):
    pos = np.random.uniform(-1, 1, 3)
    scale = np.random.uniform(0.05, 0.25, 3)
    rot = np.random.uniform(-1, 1, 4)
    rot = rot / np.linalg.norm(rot)
    f_dc = np.random.uniform(0.2, 1.0, 3)
    opacity = np.random.uniform(0.7, 1.0)
    sh = np.random.uniform(-0.1, 0.1, 45)
    vals = [*pos, *scale, *rot, *f_dc, opacity, *sh]
    lines.append(" ".join(map(str, vals)))

with open("sample100_gaussian.ply", "w") as f:
    f.write("\n".join(lines))
