#%%
import sys

sys.path.append("..")

import numpy as np

import analysis as an
import matplotlib.pyplot as plt
import cv2
import deeptrack as dt
from PIL import Image
import torch
from pytorch.unet_model import UNet

original_image = np.array(Image.open("../assets/noctiluca_image2.jpg"))
exp_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


pos = [750, 450]

# downsample
downsample = 2
exp_img = dt.Value(exp_image) >> dt.AveragePooling(ksize=(downsample, downsample))
original_image = dt.Value(original_image) >> dt.AveragePooling(
    ksize=(downsample, downsample, 1)
)
exp_img = exp_img.update()()
original_image = original_image.update()()


cropped_image = an.cropped_image(
    exp_img, [pos[0] // downsample, pos[1] // downsample], window=128
)
original_image = an.cropped_image(
    original_image, [pos[0] // downsample, pos[1] // downsample], window=128
)
# plt.imshow(cropped_image, cmap="gray")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet(
    input_shape=(1, 1, 256, 256),
    number_of_output_channels=3,  # 2 for binary segmentation and 3 for multiclass segmentation
    conv_layer_dimensions=(8, 16, 32, 64, 128, 256),  # smaller UNet (faster training)
    # conv_layer_dimensions=(64, 128, 256, 512, 1024),  # standard UNet
)

MODEL_NAME = "UNet-noctiluca-2023-02-07"

PATH = (
    "/Users/xbacss/Desktop/copepod-counter-data/models/"
    + MODEL_NAME
    + "/"
    + MODEL_NAME
    + ".pth"
)

model.load_state_dict(
    torch.load(
        PATH,
        map_location=torch.device("cpu"),
    )
)

# %%
input = cropped_image[None, None, :, :]
# input = exp_image[None, None, :, :]
input = np.array(input.to_numpy(), dtype=np.float32)
with torch.no_grad():
    output = model(torch.from_numpy(input).float().to(device))

output = torch.softmax(output, dim=1)
output = output.cpu().numpy()
output.shape
# %%
plt.figure(figsize=(10, 10))
plt.subplot(1, 4, 1)
plt.imshow(input[0, 0, :, :], cmap="gray")
plt.subplot(1, 4, 2)
plt.imshow(output[0, 0, :, :], cmap="gray")
plt.subplot(1, 4, 3)
plt.imshow(output[0, 1, :, :], cmap="gray")
plt.subplot(1, 4, 4)
plt.imshow(output[0, 2, :, :], cmap="gray")
plt.show()

# %%
fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Original image")
plt.imshow(original_image.astype(np.uint8))
plt.axis("off")
plt.subplot(1, 3, 2)
plt.title("Noctiluca segmentation")
plt.imshow(output[0, 1, :, :] > 0.5, cmap="plasma", alpha=0.8)
plt.axis("off")
plt.subplot(1, 3, 3)
plt.title("Dunaliella segmentation")
plt.imshow(output[0, 2, :, :] > 0.5, cmap="plasma", alpha=0.8)
plt.axis("off")
plt.show()
# fig.savefig("../results/prediction.jpeg", dpi=300, bbox_inches="tight")


# %%
# Save figures

fig = plt.figure(figsize=(10, 10))
plt.imshow(output[0, 2, :, :] > 0.5, cmap="gray", alpha=1)
plt.axis("off")
fig.savefig(
    "/Users/xbacss/Desktop/roadmap-planktons/figures/figure2/dunaliella_segmentation.jpeg",
    bbox_inches="tight",
    dpi=300,
)
# %%
positions = an.detect_blobs_area(output[0, 1, :, :] > 0.5, min_area=10)

fig = plt.figure(figsize=(10, 10))
plt.imshow(original_image.astype(np.uint8))
[
    plt.plot(
        p[1],
        p[0],
        "o",
        ms=300,
        markerfacecolor="None",
        markeredgecolor="green",
        markeredgewidth=6,
        alpha=1,
    )
    for p in positions
]
plt.axis("off")
plt.show()
fig.savefig(
    "/Users/xbacss/Desktop/roadmap-planktons/figures/figure2/noctiluca_detection.jpeg",
    bbox_inches="tight",
    dpi=300,
)
# %%
positions = an.detect_blobs_area(output[0, 2, :, :] > 0.5, min_area=10)

fig = plt.figure(figsize=(10, 10))
plt.imshow(original_image.astype(np.uint8))
[
    plt.plot(
        p[1],
        p[0],
        "o",
        ms=40,
        markerfacecolor="None",
        markeredgecolor="red",
        markeredgewidth=3,
        alpha=1,
    )
    for p in positions
]
plt.axis("off")
plt.show()
fig.savefig(
    "/Users/xbacss/Desktop/roadmap-planktons/figures/figure2/dunaliella_detection.jpeg",
    bbox_inches="tight",
    dpi=300,
)
# %%
