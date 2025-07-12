#!/usr/bin/python3

import numpy as np

"""
In part1.py，Implement the following functions：
def create_Gaussian_kernel(cutoff_frequency)
def my_imfilter(image, filter)
def create_hybrid_image(image1, image2, filter)
Test in proj1.ipynb

Go to the Code and notebook

implement the following functions in part1.py
def my_imfilter(image, filter):
{
} 
Potentially useful NumPy (Python library) functions: 
import numpy as np
np.pad(), which does many kinds of image padding for you；
np.clip(), which “clips” out any values in an array outside of a specified range；
np.sum() and np.multiply() , which makes it efficient to do the convolution (dot product) between the filter and windows of the image.
ndarray.shape, return the dimension; ndarray[10:20, 10:20, 1], array slice
Test in proj1_test_filtering.ipynb


"""
def create_Gaussian_kernel(cutoff_frequency):
    #创建高斯核
    """
        Returns a 2D Gaussian kernel using the specified filter size standard
        deviation and cutoff frequency.

        The kernel should have:
        - shape (k, k) where k = cutoff_frequency * 4 + 1
        - mean = floor(k / 2)
        - standard deviation = cutoff_frequency
        - values that sum to 1

        Args:
        - cutoff_frequency: an int controlling how much low frequency to leave in
          the image.
        Returns:
        - kernel: numpy nd-array of shape (k, k)

        HINT:
        - The 2D Gaussian kernel here can be calculated as the outer product of two
          vectors with values populated from evaluating the 1D Gaussian PDF at each
          corrdinate.
    """
    # 确定核的大小 k，确保是奇数
    k = cutoff_frequency * 4 + 1
    
    # 创建一个1D坐标向量，中心为0
    # 例如，k=5, cutoff_frequency=1, indices will be [-2, -1, 0, 1, 2]
    indices = np.linspace(-cutoff_frequency*2, cutoff_frequency*2, k)
    
    # 设置高斯函数的标准差
    sigma = cutoff_frequency
    
    # 计算1D高斯函数值
    # G(x) = exp(-x^2 / (2*sigma^2))
    kernel_1d = np.exp(-(indices**2) / (2 * sigma**2))
    
    # 使用外积从1D核生成2D核
    # G(x, y) = G(x) * G(y)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # 归一化，使所有元素的和为1
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    
    return kernel_2d

def my_imfilter(image, filter):
    #循环遍历像素和颜色通道进行滤波
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of shape (m, n, c)
    - filter: numpy nd-array of shape (k, j)
    Returns
    - filtered_image: numpy nd-array of shape (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using OpenCV or similar to do the
    filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that the TAs can verify
    your code works.
    """
    # 确保图像和滤波器是浮点数类型，以防计算中出现精度问题
    image = image.astype(np.float32)
    filter = filter.astype(np.float32)

    # 获取图像和滤波器的尺寸
    img_h, img_w, img_c = image.shape
    filt_h, filt_w = filter.shape

    # 计算填充的尺寸（滤波器半径）
    pad_h = filt_h // 2
    pad_w = filt_w // 2

    # 使用'symmetric'模式对图像进行填充，这种方式在边缘效果较好
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'symmetric')

    # 创建一个与原图同样大小的全零数组，用于存放结果
    filtered_image = np.zeros_like(image)

    # 遍历图像的每个颜色通道
    for c in range(img_c):
        # 遍历图像的每个像素 (y, x)
        for y in range(img_h):
            for x in range(img_w):
                # 从填充后的图像中提取与滤波器同样大小的区域
                # 当前中心点 (y,x) 对应到填充图中的 (y+pad_h, x+pad_w)
                region = padded_image[y : y + filt_h, x : x + filt_w, c]
                
                # 执行卷积操作：将区域与滤波器逐元素相乘，然后求和
                # 这就是加权平均的过程
                pixel_value = np.sum(region * filter)
                
                # 将计算得到的值赋给输出图像的相应位置
                filtered_image[y, x, c] = pixel_value
    
    return filtered_image

def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (x, y)
    Returns
    - low_frequencies: numpy nd-array of shape (m, n, c)
    - high_frequencies: numpy nd-array of shape (m, n, c)
    - hybrid_image: numpy nd-array of shape (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are between
      0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """
    # 确保图像为浮点类型
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # 1. 生成低频图像：对 image1 进行低通滤波
    low_frequencies = my_imfilter(image1, filter)

    # 2. 生成高频图像：从 image2 中减去它的低频部分
    #    首先获取 image2 的低频部分
    low_pass_image2 = my_imfilter(image2, filter)
    #    然后用原图减去低频部分得到高频部分
    high_frequencies = image2 - low_pass_image2
    
    # 3. 创建混合图像：将 image1 的低频和 image2 的高频相加
    hybrid_image = low_frequencies + high_frequencies
    
    # 4. 裁剪像素值：确保最终图像的像素值在 [0, 1] 范围内
    #    这对于显示和保存图像至关重要
    hybrid_image = np.clip(hybrid_image, 0.0, 1.0)
    
    # 返回所有三个结果图像
    return low_frequencies, high_frequencies, hybrid_image