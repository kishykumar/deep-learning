from torchvision import datasets
import os

# this will auto-download the ubyte files if you don't already have them
dataset = datasets.MNIST(root="./data", train=False, download=True)

os.makedirs("mnist_pngs", exist_ok=True)

# save the first 20 test images as PNGs
for i in range(20):
    img, label = dataset[i]  # img is already a PIL.Image
    img.save(f"mnist_pngs/{i}_label{label}.png")