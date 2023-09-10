#%%
import sys

sys.path.append("..")

import numpy as np

# import analysis as an
import matplotlib.pyplot as plt
import cv2
import deeptrack as dt
from PIL import Image

#%%


# exp_image = np.array(Image.open("../assets/noctiluca_image2.jpg"))
# exp_image = cv2.cvtColor(exp_image, cv2.COLOR_BGR2GRAY)

# pos = [750, 450]

# # downsample
# downsample = 2
# exp_img = dt.Value(exp_image) >> dt.AveragePooling(ksize=(downsample, downsample))
# exp_img = exp_img.update()()
# print(exp_img.shape)
# plt.imshow(exp_img, cmap="gray")
# plt.show()

# cropped_image = an.cropped_image(
#     exp_img, [pos[0] // downsample, pos[1] // downsample], window=128
# )
# plt.imshow(cropped_image, cmap="gray")

IMAGE_SIZE = 256  # + 32

optics = dt.Fluorescence(
    wavelength=500e-9,
    NA=1.2,
    resolution=1e-6,
    magnification=12,
    refractive_index_medium=1.33,
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
)


# point_particles = dt.PointParticle(
#     position=lambda: np.random.uniform(0, IMAGE_SIZE, 2),
#     intensity=lambda: np.random.uniform(200, 300),
#     z=lambda: np.random.uniform(-5, 5),
# )

point_particles = dt.Sphere(
    position=lambda: np.random.uniform(0, IMAGE_SIZE, 2),
    radius=lambda: np.random.uniform(0.2e-6, 0.4e-6),
    intensity=lambda: np.random.uniform(1, 1.5),
    z=lambda: np.random.uniform(-5, 5),
)

inner_spheres = dt.Sphere(
    position=lambda: np.random.uniform(0, IMAGE_SIZE, 2),
    radius=lambda: np.random.uniform(2e-6, 5e-6),
    intensity=lambda: -1 * np.random.uniform(0.8, 1.2),
)

outer_spheres = dt.Sphere(
    position=inner_spheres.position,
    radius=inner_spheres.radius * 1.1,
    intensity=inner_spheres.intensity * -1,
)

# combined_spheres = outer_spheres >> inner_spheres
combined_spheres = inner_spheres >> outer_spheres

point_particles_in_image = lambda: np.random.randint(20, 30)
spheres_in_image = lambda: np.random.randint(1, 3)


point_cells = (
    (point_particles ^ point_particles_in_image)
    >> dt.Pad(px=(5, 5, 5, 5))
    >> dt.ElasticTransformation(alpha=50, sigma=8, order=1)
    >> dt.CropTight()
    >> dt.Poisson(snr=3)
    # You can add more transformations here
)

spherical_cells = (
    (combined_spheres ^ spheres_in_image)
    >> dt.Pad(px=(5, 5, 5, 5))
    >> dt.ElasticTransformation(alpha=50, sigma=8, order=1)
    >> dt.CropTight()
    >> dt.Poisson(snr=3)
    # You can add more transformations here
)

normalization = dt.NormalizeMinMax(
    min=lambda: np.random.rand() * 0.4,
    max=lambda min: min + 0.1 + np.random.rand() * 0.5,
)
noise = dt.Poisson(snr=lambda: np.random.uniform(30, 40), background=normalization.min)

# sample = optics(dt.NonOverlapping(point_cells & spherical_cells)) >> normalization >> noise
sample = optics(point_cells & spherical_cells) >> normalization >> noise


def transf():
    def inner(scatter_mask):

        mask = scatter_mask.sum(-1) != 0
        output = np.zeros((*scatter_mask.shape[:2], 1))

        output[mask] = 1

        return output

    return inner


def transf2(circle_radius=3):
    def inner(image):
        X, Y = np.mgrid[: 2 * circle_radius, : 2 * circle_radius]
        CIRCLE = (X - circle_radius + 0.5) ** 2 + (
            Y - circle_radius + 0.5
        ) ** 2 <= circle_radius**2
        CIRCLE = CIRCLE[..., None]
        return CIRCLE

    return inner


masks1 = spherical_cells >> dt.SampleToMasks(
    transf, output_region=optics.output_region, merge_method="or", number_of_masks=1
)

masks2 = point_cells >> dt.SampleToMasks(
    transf2, output_region=optics.output_region, merge_method="or", number_of_masks=1
)

image_and_labels = sample & masks1 & masks2

# Augmentations
# image_and_labels = (
#     dt.Reuse(image_and_labels, uses=8)
#     >> dt.FlipUD()
#     >> dt.FlipLR()
#     >> dt.FlipDiagonal()
# )
#%%

def generate_images():
    return image_and_labels


# #%%
im, m1, m2 = generate_images().update()()

# plt.imshow(im, cmap="gray")

fig = plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(im, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(m1[..., 0], cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(m2[..., 0], cmap="gray")
plt.show()

#%%
fig = plt.figure(figsize=(10, 10))
plt.imshow(m2[..., 0], cmap="gray", alpha=1)
plt.axis("off")
fig.savefig(
    "/Users/xbacss/Desktop/roadmap-planktons/figures/figure2/simulation/duna_segmentation.jpeg",
    bbox_inches="tight",
    dpi=300,
)

# %%
