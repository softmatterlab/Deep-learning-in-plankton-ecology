"""
Author: Harshith Bachimanchi
Date: 2023-09-10
Description: UNet model and other utilities
"""

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


def load_video(path, start=0, end=None):
    video = cv2.VideoCapture(path)
    frames = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end:
        end = min(end, total_frames)
    else:
        end = total_frames
    for i in tqdm(range(start, end, 1), desc="loading frames"):
        video.set(1, i)
        ret, frame = video.read()
        gray_frame = frame  # [:,:,0]
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


import torch.nn as nn
import torch, torchvision


class DoubleConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super().__init__()
        self.relu = relu
        self.double_convolution_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.double_convolution_block_wihout_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.relu:
            x = self.double_convolution_block(x)
        else:
            x = self.double_convolution_block_wihout_relu(x)
        return x


class UNetEncoder(nn.Module):
    """
    Encoder for UNet: Halves the spatial dimensions and doubles the number of channels.
    """

    def __init__(
        self,
        input_shape=(1, 3, 572, 572),
        conv_layer_dimensions=(64, 128, 256, 512, 1024),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoder_block1 = DoubleConvolutionBlock(
            input_shape[1], conv_layer_dimensions[0]
        )
        self.encoder_blocks = nn.ModuleList(
            [
                DoubleConvolutionBlock(
                    conv_layer_dimensions[i],
                    conv_layer_dimensions[i + 1],
                )
                for i in range(len(conv_layer_dimensions) - 1)
            ]
        )
        self.pool = nn.MaxPool2d(2, padding=0)

    def forward(self, x):
        feature_maps = []
        x = self.encoder_block1(x)
        feature_maps.append(x)
        x = self.pool(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            feature_maps.append(x)
            x = self.pool(x)
        return feature_maps


class UNetDecoder(nn.Module):
    """
    Decoder for UNet: Doubles the spatial dimensions and halves the number of channels.
    """

    def __init__(self, conv_layer_dimensions=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.conv_layer_dimensions = conv_layer_dimensions
        # upconvolution layers
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    conv_layer_dimensions[i],
                    conv_layer_dimensions[i + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
                for i in range(len(conv_layer_dimensions) - 1)
            ]
        )
        # double convolution layers after upconvolution
        self.decoder_blocks = nn.ModuleList(
            [
                DoubleConvolutionBlock(
                    conv_layer_dimensions[i],
                    conv_layer_dimensions[i + 1],
                )
                for i in range(len(conv_layer_dimensions) - 2)
            ]
        )
        # final double convolution layer should not have relu
        self.last_decoder_block = DoubleConvolutionBlock(
            conv_layer_dimensions[-2], conv_layer_dimensions[-1], relu=False
        )

    def forward(self, x, feature_maps):
        for i in range(len(self.conv_layer_dimensions) - 1):
            x = self.upconvs[i](x)
            features_maps_crops = self.crop(feature_maps[i], x)
            x = torch.cat([x, features_maps_crops], dim=1)
            if i == len(self.conv_layer_dimensions) - 2:  # when i = 3 i.e, 128
                x = self.last_decoder_block(x)
            else:
                x = self.decoder_blocks[i](x)
        return x

    def crop(self, feature_maps, x):
        _, _, H, W = x.shape
        cropped_feature_maps = torchvision.transforms.CenterCrop((H, W))(feature_maps)
        cropped_feature_maps = feature_maps
        return cropped_feature_maps


class UNet(nn.Module):
    def __init__(
        self,
        input_shape=(1, 1, 128, 128),
        conv_layer_dimensions=(16, 32, 64, 128, 256),
        number_of_output_channels=1,
    ):
        super().__init__()
        self.encoder = UNetEncoder(
            input_shape=input_shape, conv_layer_dimensions=conv_layer_dimensions
        )
        self.decoder = UNetDecoder(
            conv_layer_dimensions=conv_layer_dimensions[::-1]
        )  # reverse the list
        self.final_conv = nn.Conv2d(
            conv_layer_dimensions[0], number_of_output_channels, 3, padding=1
        )

    def forward(self, x):
        feature_maps = self.encoder(x)
        x = self.decoder(feature_maps[-1], feature_maps[::-1][1:])
        x = self.final_conv(x)
        return x
