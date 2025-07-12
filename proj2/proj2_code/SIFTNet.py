#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from torchvision import transforms as transforms
from torch import nn
from proj2_code.torch_layer_utils import ImageGradientsLayer


"""
Authors: John Lambert, Vijay Upadhya, Patsorn Sangkloy, Cusuh Ham,
Frank Dellaert, September 2019.
...
"""


class HistogramLayer(nn.Module):
    def __init__(self) -> None:
        """
        初始化一个无参数的直方图层。
        """
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        """
        在每个像素位置上形成一个加权直方图。
        """
        cosines = x[:,:8,:,:]
        im_grads = x[:,8:,:,:]

        per_px_histogram = torch.zeros_like(cosines)
        max_idx = torch.argmax(cosines, dim=1)
        magnitudes = torch.norm(im_grads, dim=1)
        per_px_histogram.scatter_(1, max_idx.unsqueeze(1), magnitudes.unsqueeze(1))
        
        return per_px_histogram

class SubGridAccumulationLayer(nn.Module):
    """
    子网格累加层。
    """
    def __init__(self) -> None:
        """
        初始化一个特殊的卷积层用于在4x4窗口内累加特征。
        """
        super().__init__()
        self.layer = nn.Conv2d(in_channels=8,
                                 out_channels=8,
                                 kernel_size=4,
                                 stride=1,
                                 padding=2,
                                 groups=8,
                                 bias=False)
        self.layer.weight.data.fill_(1.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行子网格累加的前向传播。
        """
        return self.layer(x)


def angles_to_vectors_2d_pytorch(angles: torch.Tensor) -> torch.Tensor:
    """
    将一批弧度制角度转换为2D单位向量。
    """
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    return torch.stack([cosines, sines], dim=1)


class SIFTOrientationLayer(nn.Module):
    """
    SIFT方向层，将图像梯度投影到8个预定义的方向上。
    """
    def __init__(self):
        """
        初始化1x1卷积层并设置其权重。
        """
        super().__init__()
        self.layer = nn.Conv2d(in_channels=2,
                                 out_channels=10,
                                 kernel_size=1,
                                 bias=False)
        self.layer.weight = self.get_orientation_bin_weights()

    def get_orientation_bin_weights(self) -> torch.nn.Parameter:
        """
        构建1x1卷积层的权重，用于计算梯度和8个方向的点积。
        """
        # 使用与原始模板完全相同的逻辑来生成角度，以匹配单元测试
        ratios = torch.arange(start=0.125, end=2.125, step=0.25)
        orientation_radians = ratios * np.pi
        orientation_vectors = angles_to_vectors_2d_pytorch(orientation_radians)

        identity_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.cat((orientation_vectors, identity_vectors), dim=0)
        
        return nn.Parameter(weights.unsqueeze(-1).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过1x1卷积实现梯度到方向的投影。
        """
        return self.layer(x)


class SIFTNet(nn.Module):
    def __init__(self):
        """
        将各个层组合成一个完整的SIFT网络。
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            ImageGradientsLayer(),
            SIFTOrientationLayer(),
            HistogramLayer(),
            SubGridAccumulationLayer()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行SIFTNet的前向传播。
        """
        return self.net(x)


def get_sift_subgrid_coords(x_center: int, y_center: int):
    """
    计算一个16x16区域内，16个4x4子网格的中心采样坐标。
    """
    x_offsets = np.array([-6, -2, 2, 6])
    y_offsets = np.array([-6, -2, 2, 6])
    x_grid = np.tile(x_center + x_offsets, 4)
    y_grid = np.repeat(y_center + y_offsets, 4)
    return x_grid, y_grid


def get_siftnet_features(img_bw: torch.Tensor, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    根据给定的(x,y)关键点坐标，提取SIFT特征。
    """
    assert img_bw.shape[0] == 1
    assert img_bw.shape[1] == 1
    
    if img_bw.dtype != torch.float32:
        img_bw = img_bw.float()

    net = SIFTNet()
    features = net(img_bw).squeeze(0)
    
    K = x.shape[0] 
    fvs = torch.zeros([K, 128])

    for i in range(K):
        x_coords, y_coords = get_sift_subgrid_coords(int(x[i]), int(y[i]))
        
        if np.any(x_coords < 0) or np.any(x_coords >= features.shape[2]) or \
           np.any(y_coords < 0) or np.any(y_coords >= features.shape[1]):
            continue

        img_features = features[:, y_coords, x_coords].flatten()
        
        norm = torch.linalg.norm(img_features)
        if norm > 1e-6:
            img_features = img_features / norm
        
        fvs[i] = torch.pow(img_features, 0.9)
        
    return fvs.detach().cpu().numpy()