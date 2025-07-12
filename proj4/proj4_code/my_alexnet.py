import torch
import torch.nn as nn

from torchvision.models import alexnet

#修改一下结构，原来的太redundant
class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function
    '''
    super().__init__()

    # 1. 加载预训练的AlexNet模型
    pretrained_alexnet = alexnet(pretrained=True)

    # 2. 拆分并冻结卷积层
    self.cnn_layers = pretrained_alexnet.features
    for param in self.cnn_layers.parameters():
        param.requires_grad = False

    # 3. 提取平均池化层
    self.avgpool = pretrained_alexnet.avgpool

    # 4. 拆分、修改并精确冻结全连接层
    self.fc_layers = pretrained_alexnet.classifier
    
    # 冻结前两个线性层，它们的索引是 1 和 4
    for i in [1, 4]:
        for param in self.fc_layers[i].parameters():
            param.requires_grad = False
            
    # 替换最后一层以匹配类别数 (15)
    num_features = self.fc_layers[6].in_features
    self.fc_layers[6] = nn.Linear(num_features, 15)

    # 5. 定义损失函数
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net
    '''
    x = self.cnn_layers(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    y = self.fc_layers(x)
    
    return y