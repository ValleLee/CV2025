'''
Utilities to be used along with the deep model
'''

import torch


def predict_labels(model: torch.nn.Module, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass and extract the labels from the model output

    Args:
    -   model: a model (which inherits from nn.Module)
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   predicted_labels: the output labels [Dim: (N,)]
    '''
    
    # 将模型设置为评估模式，这会关闭 Dropout 等层
    model.eval()
    
    # 在不计算梯度的上下文中执行，以节省计算资源
    with torch.no_grad():
        # 1. 通过模型前向传播，得到原始得分 (raw scores)
        scores = model(x)
        
        # 2. 找到每个样本得分最高的那个类别的索引，作为预测标签
        #    dim=1 表示在“类别”这个维度上寻找最大值
        predicted_labels = torch.argmax(scores, dim=1)
        
    return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
    '''
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    '''

    # 使用模型内部定义的损失函数 (我们在 SimpleNet 中定义了它)
    loss = model.loss_criterion(model_output, target_labels)

    # 如果需要归一化，则将总损失除以批次大小，得到平均损失
    if is_normalize:
        batch_size = model_output.shape[0]
        loss = loss / batch_size
        
    return loss