#!/usr/bin/python3

from torch import nn
import torch
from typing import Tuple
import numpy as np
import torch.nn.functional as F


from proj2_code.torch_layer_utils import (
    get_sobel_xy_parameters,
    get_gaussian_kernel,
    ImageGradientsLayer
)

"""
Authors: Patsorn Sangkloy, Vijay Upadhya, John Lambert, Cusuh Ham,
Frank Dellaert, September 2019.
"""


class HarrisNet(nn.Module):
    """
    Implement Harris corner detector (See Szeliski 4.1.1) in pytorch by
    sequentially stacking several layers together.
    """

    def __init__(self):
        """
        Create a nn.Sequential() network...
        """
        super().__init__()

        # 按照Harris角点检测的流程，正确排序各个网络层。
        image_gradients_layer = ImageGradientsLayer()
        channel_product_layer = ChannelProductLayer()
        second_moment_matrix_layer = SecondMomentMatrixLayer()
        corner_response_layer = CornerResponseLayer()
        nms_layer = NMSLayer()

        self.net = torch.nn.Sequential(
            image_gradients_layer,
            channel_product_layer,
            second_moment_matrix_layer,
            corner_response_layer,
            nms_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of HarrisNet network...
        """
        assert x.dim() == 4, \
            "Input should have 4 dimensions. Was {}".format(x.dim())

        return self.net(x)


class ChannelProductLayer(nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy...
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer...
        """
        num_image, c, height, width = x.shape
        output = torch.zeros([num_image, 3, height, width], device=x.device)

        for i in range(num_image):
            image_gx = x[i, 0, :, :]
            image_gy = x[i, 1, :, :]
            
            I_xx = torch.mul(image_gx, image_gx)
            I_yy = torch.mul(image_gy, image_gy)
            I_xy = torch.mul(image_gx, image_gy)
            
            output[i] = torch.stack([I_xx, I_yy, I_xy], dim=0)
            
        return output


class SecondMomentMatrixLayer(nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image...
    """
    def __init__(self, ksize: int = 7, sigma: float = 5.0):
        """
        You may find get_gaussian_kernel() useful...
        """
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        
        self.kernel = get_gaussian_kernel(self.ksize, self.sigma).unsqueeze(0).unsqueeze(0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer...
        """
        num_image, c, height, width = x.shape
        output = torch.zeros([num_image, 3, height, width], device=x.device)
        kernel = self.kernel.to(x.device, dtype=x.dtype)
        
        for i in range(num_image):
            I_xx = x[i, 0, :, :].unsqueeze(0).unsqueeze(0)
            I_yy = x[i, 1, :, :].unsqueeze(0).unsqueeze(0)
            I_xy = x[i, 2, :, :].unsqueeze(0).unsqueeze(0)

            S_xx = F.conv2d(input=I_xx, weight=kernel, padding=self.ksize//2)
            S_yy = F.conv2d(input=I_yy, weight=kernel, padding=self.ksize//2)
            S_xy = F.conv2d(input=I_xy, weight=kernel, padding=self.ksize//2)

            output[i, 0, :, :] = S_xx
            output[i, 1, :, :] = S_yy
            output[i, 2, :, :] = S_xy
            
        return output


class CornerResponseLayer(nn.Module):
    """
    Compute R matrix...
    """
    def __init__(self, alpha: float=0.05):
        """
        Don't modify this __init__ function!
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R...
        """
        num_image, c, height, width = x.shape
        output = torch.zeros([num_image, 1, height, width], device=x.device)
        
        for i in range(num_image):
            S_xx = x[i, 0, :, :]
            S_yy = x[i, 1, :, :]
            S_xy = x[i, 2, :, :]
            
            det = torch.mul(S_xx, S_yy) - torch.mul(S_xy, S_xy)
            trace = S_xx + S_yy
            
            R = det - self.alpha * (trace ** 2)
            output[i] = R
            
        return output


class NMSLayer(nn.Module):
    """
    NMSLayer: Perform non-maximum suppression...
    """
    def __init__(self, ksize: int = 7):
        """
        Initializes the NMSLayer.
        """
        super().__init__()
        self.ksize = ksize
        self.pool = nn.MaxPool2d(kernel_size=self.ksize, stride=1, padding=self.ksize//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform non-maximum suppression...
        """
        maximums = self.pool(x)
        is_max = (x == maximums)
        return x * is_max


def get_interest_points(image: torch.Tensor, num_points: int = 4500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to return top most N x,y points...
    """
    harris_detector = HarrisNet()
    R = harris_detector(image)

    _, _, y, x = torch.nonzero(R, as_tuple=True)
    confidences = R[0, 0, y, x]
    
    indices = torch.argsort(confidences, descending=True)
    
    x = x[indices]
    y = y[indices]
    confidences = confidences[indices]

    if len(x) > num_points:
        x = x[:num_points]
        y = y[:num_points]
        confidences = confidences[:num_points]
    
    x, y, confidences = remove_border_vals(image, x, y, confidences)
    return x, y, confidences


def remove_border_vals(img, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, border: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove interest points that are too close to a border...
    """
    height, width = img.shape[-2:]
    
    y_valid = (y >= border) & (y < height - border)
    x_valid = (x >= border) & (x < width - border)
    valid_mask = y_valid & x_valid

    return x[valid_mask], y[valid_mask], c[valid_mask]