import os
from pickletools import optimize

from PIL import Image

folder = input("Enter folder path: ").strip().strip('"')

# get all PNG files
files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
if not files:
    print("No PNG files found!")
    exit()

# sort ascending
files.sort()

# load images
images = [Image.open(os.path.join(folder, f)) for f in files]

# save GIF
output_path = os.path.join(folder, "output.gif")
images[0].save(
    output_path,
    save_all=True,
    append_images=images[1:],
    duration=500,  # ms per frame (100 ms = 10 fps)
    loop=0,
    optimize=True,
)

print(f"GIF created: {output_path}")
