import torch
import torch.nn as nn


class SimpleNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
    '''
    super().__init__()

    # 卷积模块(cnn_layers): 包含2个'卷积-激活-池化'层，用于提取图像特征。
    # 输入: (N, 3, 64, 64) -> 输出: (N, 16, 16, 16)
    self.cnn_layers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    # 全连接分类模块(fc_layers): 接收展平后的特征并进行分类。
    # 输入: 16*16*16=4096 -> 输出: 15 (对应15个类别)
    self.fc_layers = nn.Sequential(
        nn.Linear(in_features=16 * 16 * 16, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=15)
    )
    
    # 定义损失函数，并根据要求使用 'sum' reduction
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    
    # 1. 数据通过卷积模块
    x = self.cnn_layers(x)
    
    # 2. 展平特征图 (Flatten)，为全连接层做准备
    x = torch.flatten(x, 1)
    
    # 3. 数据通过全连接模块，得到分类得分
    y = self.fc_layers(x)
    
    return y