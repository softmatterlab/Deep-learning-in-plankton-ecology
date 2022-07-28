import sys
import os

sys.path.append("..")
sys.path.append(".")

import random
import numpy as np
import cv2
from tqdm import tqdm
import datetime
import deeptrack as dt


date = str(datetime.date.today())


video_path = "./videos/MVI_5969.MOV"
model_name = "planktons_lodestar_" + str(date)


os.makedirs("/data/planktons-lodestar/" + model_name + "/", exist_ok=True)
model_save_path = "/data/planktons-lodestar/" + model_name + "/" + model_name + ".h5"
# history_save_path = "/data/planktons-lodestar/"  + model_name + "/" + model_name + "_history.png"
# info_save_path = "/data/planktons-lodestar/"  + model_name + "/" + model_name + "_info.txt"


def load_video(path, start=0, end=100):
    video = cv2.VideoCapture(path)
    frames = []
    for i in tqdm(range(start, end), desc="loading frames"):
        video.set(1, i)
        ret, frame = video.read()
        gray_frame = frame  # [:,:,0]
        frames.append(gray_frame)
    return np.array(frames)


frames = load_video(video_path)

frame = frames[0]

plankton_positions = [
    [1005, 375],
    [1139, 198],
    [166, 157],
    [1575, 511],
    [525, 222],
    [408, 690],
]

crop_width = 150
crops = []
for pos in plankton_positions:
    crop = frame[
        int(pos[1] - crop_width / 2) : int(pos[1] + crop_width / 2),
        int(pos[0] - crop_width / 2) : int(pos[0] + crop_width / 2),
    ][:, :, 0]
    crops.append(crop)
crops = np.expand_dims(crops, axis=-1)


random_crop = dt.Value(lambda: random.choice(crops))

downsample = 2

model = dt.models.LodeSTAR(input_shape=(None, None, 1))
train_set = dt.Value(random_crop) >> dt.AveragePooling(
    ksize=(downsample, downsample, 1)
)
train_set.plot(cmap="gray")

model.fit(train_set, epochs=10, batch_size=8)

model.summary

model.save(model_save_path)
