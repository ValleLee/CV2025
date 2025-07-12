'''
Contains functions with different data transforms
'''

import numpy as np
import torchvision.transforms as transforms

from typing import Tuple


def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
    '''
    Returns the core transforms needed to feed the images to our model

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean  of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - fundamental_transforms: transforms.Compose with the fundamental transforms
    '''

    return transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std)
    ])


def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
    '''
    Returns the data augmentation + core transforms needed to be applied on the
    train set

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean  of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - aug_transforms: transforms.Compose with all the transforms
    '''

    # 数据增强转换列表，在基础转换之前应用
    augmentation_transforms = [
        # 对应任务要求的 "mirror" (镜像)，随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 对应任务要求的 "jitter" (抖动)，随机改变图像的色彩属性
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # 增加一个常用的旋转变换，让模型对物体角度更不敏感
        transforms.RandomRotation(15),
    ]
    
    # 基础转换列表，确保图像尺寸、类型和分布符合模型要求
    fundamental_transforms = [
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std)
    ]
    
    # 将增强转换和基础转换合并成一个流水线
    # 顺序至关重要：先做各种随机变换，最后再统一尺寸和归一化
    return transforms.Compose(augmentation_transforms + fundamental_transforms)