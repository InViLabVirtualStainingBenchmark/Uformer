import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Paths
noisy_dir = "datasets/denoising/sidd/val/input"
denoised_dir = "results/sidd/png"

# Pick a sample patch (you can change this)
patch = "0001_01.png"

noisy_path = os.path.join(noisy_dir, patch)
denoised_path = os.path.join(denoised_dir, patch)

print("Noisy:", noisy_path)
print("Denoised:", denoised_path)

noisy = np.array(Image.open(noisy_path))
denoised = np.array(Image.open(denoised_path))

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Noisy Input")
plt.imshow(noisy)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Denoised (Uformer)")
plt.imshow(denoised)
plt.axis("off")

plt.show()

