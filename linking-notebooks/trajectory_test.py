#%%
import sys

sys.path.append("..")
import numpy as np
import analysis as an
from utils import load_video
import cv2


def color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


END = 30
# Load plankton positions
pos = np.load("../detections/lodestar-detections-plankton2.npy", allow_pickle=True)
pos = pos[:END]
pos.shape
# Load video
frames = load_video(
    "/Users/xbacss/Documents/GitHub/Plankton-Review/original-videos/Plankton2.mp4",
    end=END,
)
# %%
traj_data = an.get_trajectory_data(pos, minimum_length=10)
# %%

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(18, 12))
ax.imshow(color(frames[-1]), cmap="gray")
plt.axis("off")
plt.scatter(
    pos[-1][:, 1],
    pos[-1][:, 0],
    color="red",
    s=600,
    facecolors="none",
    edgecolors="darkred",
    linewidth=2,
)
i = 0
for trajectory in traj_data:
    ax.plot(trajectory[:, 2], trajectory[:, 1], "orange", alpha=1, linewidth=2)
    # ax.text(trajectory[:,2][-1], trajectory[:,1][-1], str(i), color="white", fontsize=16)
    i += 1

plt.savefig(
    "/Users/xbacss/Documents/Courses/ScientificWriting/figures/figure3/trajectory.png",
    dpi=300,
    bbox_inches="tight",
)
# %%

fig = plt.figure(figsize=(18, 12))
plt.imshow(color(frames[-1]), cmap="gray")

plt.axis("off")
fig.savefig(
    "/Users/xbacss/Documents/Courses/ScientificWriting/figures/figure3/original.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
