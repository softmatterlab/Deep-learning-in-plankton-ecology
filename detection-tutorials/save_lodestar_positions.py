# %%
# This code is not part of the tutorial, but is used to save the positions obtained from the LodeSTAR model, for trajectory linking application.

import deeptrack as dt
import numpy as np

from utils import load_video
import matplotlib.pyplot as plt

PATH = "./original-videos/Plankton1.mp4"

video_frames = load_video(PATH, start=0, end=100)
# %%
# Load pre-trained model
model = dt.models.LodeSTAR(
    input_shape=(None, None, 1),
)
model.model.model.load_weights("./pre-trained-models/lodestar-model-plankton1.h5")

# %%
# Predict positions
# Test for one frame
alpha = 0.1
cutoff = 0.1
downsample = 2

test_frame = video_frames[0][:, :, 0]
test_frame = test_frame.reshape(1, *test_frame.shape, 1)
test_frame.shape

a, b = model.predict(test_frame)
# %%
plt.imshow(np.log(b[0, :, :, 0]), cmap="gray")
plt.show()

plt.imshow(a[0, :, :, 0] * np.log(b[0, :, :, 0]), cmap="gray")
plt.show()

# detections = model.predict_and_detect(
#     test_frame[0:1, ::downsample, ::downsample, :],
#     alpha=alpha,
#     beta = 1-alpha,
#     cutoff=cutoff,
#     mode="constant",
# )
# %%
detections = np.array(detections, dtype=int) * downsample
# %%
# plot detections
import matplotlib.pyplot as plt

plt.imshow(test_frame[0, :, :, 0], cmap="gray")
[
    plt.plot(
        d[1], d[0], "o", markerfacecolor="None", markeredgecolor="r", markersize=10
    )
    for d in detections[0]
]
plt.show()
# %%
# Predict positions for all frames

frames = video_frames[:, :, :, 0][..., np.newaxis]

detections = model.predict_and_detect(
    frames[:, ::downsample, ::downsample, :],
    alpha=alpha,
    cutoff=cutoff,
    mode="constant",
)
# %%
detections = np.array(detections) * downsample
np.shape(detections)

# %%
np.std([len(d) for d in detections])

# %%
# Save detections
np.save(
    "./detections/lodestar-detections-plankton1.npy",
    detections,
)
# %%
# test loading
pos = np.load(
    "./detections/lodestar-detections-plankton1.npy",
    allow_pickle=True,
)
# %%
pos.shape
# %%
