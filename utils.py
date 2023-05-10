import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
# import deeptrack as dt
# import warnings
# warnings.filterwarnings("ignore")

def load_video(path, start=0, end=None):
    video= cv2.VideoCapture(path)
    frames=[]
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end:
        end = min(end, total_frames)
    else:
        end = total_frames
    for i in tqdm(range(start, end, 1), desc="loading frames"):
        video.set(1, i)
        ret, frame=video.read()
        gray_frame = frame#[:,:,0]
        frames.append(gray_frame)
    return np.array(frames)


def create_df_planktons(path, no_frames=[], scaling_x=1280, scaling_y=1024):
    data = np.load(path, allow_pickle=True)
    if no_frames:
        data = data[: no_frames[0]]

    detections = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            detections.append(
                [
                    (data[i][:, 0][j]) / scaling_x,
                    (data[i][:, 1][j]) / scaling_y,
                    i,
                    0,
                    0,
                    0,
                ]
            )

    dfs = pd.DataFrame(
        detections,
        index=None,
        columns=[
            "centroid-0",
            "centroid-1",
            "frame",
            "solution",
            "set",
            "label",
        ],
    )

    dfs = dfs.astype({"frame": "int32", "set": "int32"})

    return dfs


