import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
    '''
    Compute the mean and the standard deviation of the dataset.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    '''
    
    # 使用StandardScaler的partial_fit方法，可以增量计算统计数据，节省内存
    scaler = StandardScaler()

    all_image_paths = []
    # 遍历根目录下的所有子目录（如 train, test）
    for subdir_path in glob.glob(os.path.join(dir_name, '*')):
        # 确保处理的是文件夹
        if os.path.isdir(subdir_path):
            # 遍历每个子目录下的所有类别文件夹
            for cat_folder in glob.glob(os.path.join(subdir_path, '*')):
                 if os.path.isdir(cat_folder):
                    # 将该类别下所有jpg文件的路径添加到列表中
                    all_image_paths.extend(glob.glob(os.path.join(cat_folder, '*.jpg')))

    for path in all_image_paths:
        image = np.array(Image.open(path).convert('L')) / 255.0
        # 将2D图像数据展平成1D列向量，以符合partial_fit的输入要求
        image = image.reshape(-1, 1)
        scaler.partial_fit(image)
        
    mean = scaler.mean_
    std = scaler.scale_
    
    return mean, std