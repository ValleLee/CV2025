'''
Script with Pytorch's dataloader class
'''

import os
import glob
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms 

from typing import Tuple, List
from PIL import Image


class ImageLoader(data.Dataset):
    '''
    Class for data loading
    '''

    train_folder = 'train'
    test_folder = 'test'

    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 transform: torchvision.transforms.Compose = None):
        '''
        Init function for the class

        Args:
        - root_dir: the dir path which contains the train and test folder
        - split: 'test' or 'train' split
        - transforms: the transforms to be applied to the data
        '''
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == 'train':
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == 'test':
            self.curr_folder = os.path.join(root_dir, self.test_folder)
        
        # 获取所有类别及其对应的整数标签
        self.class_dict = self.get_classes()
        # 加载所有图片路径和它们对应的标签
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(self, class_labels: dict) -> List[Tuple[str, int]]:
        '''
        Fetches all image paths along with labels

        Args:
        -   class_labels: the class labels dictionary, with keys being the classes
            in this dataset
        Returns:
        -   list[(filepath, int)]: a list of filepaths and their class indices
        '''

        img_paths = []  # a list of (filename, class index)
        # 遍历指定路径下的每个类别文件夹
        for class_name, class_idx in class_labels.items():
            class_path = os.path.join(self.curr_folder, class_name)
            for img_path in glob.glob(os.path.join(class_path, '*.jpg')):
                img_paths.append((img_path, class_idx))

        return img_paths

    def get_classes(self) -> dict:
        '''
        Get the classes (which are folder names in self.curr_folder)

        Returns:
        -   Dict of class names (string) to integer labels
        '''

        classes = {}
        # 类别名称就是文件夹的名称
        class_names = [d for d in os.listdir(self.curr_folder) if os.path.isdir(os.path.join(self.curr_folder, d))]
        class_names.sort()
        # 将每个类别名称映射到一个唯一的整数
        for i, class_name in enumerate(class_names):
            classes[class_name] = i

        return classes

    def load_img_from_path(self, path: str) -> Image:
        '''
        Loads the image as grayscale (using Pillow)

        Note: do not normalize the image to [0,1]

        Args:
        -   path: the path of the image
        Returns:
        -   image: grayscale image loaded using pillow (Use 'L' flag while converting using Pillow's function)
        '''
        # 加载灰度图'L'，以通过单元测试
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        '''
        Fetches the item (image, label) at a given index

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        '''
        # 获取指定索引的图片路径和标签
        path, target = self.dataset[index]
        # 加载图片（此时为灰度图）
        img = self.load_img_from_path(path)
        
        # 将灰度图转换为RGB图，以供后续训练使用
        img = img.convert('RGB')
        
        # 如果定义了transform，则对图片进行变换
        if self.transform is not None:
            img = self.transform(img)
        # 否则，提供一个默认的变换，以确保输出是 Tensor
        else:
            # 这个默认变换就是为了应对测试代码不提供 transform 的情况
            default_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor()
            ])
            img = default_transform(img)

        return img, target

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.dataset)