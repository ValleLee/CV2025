'''
This class contains helper functions which will help get the optimizer
'''

import torch


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    '''
    Returns the optimizer initializer according to the config

    Note: config has a minimum of three entries.
    Feel free to add more entries if you want.
    But do not change the name of the three existing entries

    Args:
    - model: the model to optimize for
    - config: a dictionary containing parameters for the config
    Returns:
    - optimizer: the optimizer
    '''

    optimizer = None
    
    # 获取配置参数，并设置更合理的默认值
    optimizer_type = config.get("optimizer_type", "sgd")
    learning_rate = config.get("lr", 1e-3)           # 使用 'lr' 作为键，并提供一个合理的默认学习率
    weight_decay = config.get("weight_decay", 1e-5)  # 提供一个合理的默认权重衰减值

    # 将打印信息的逻辑集中到前面
    print(f'optimizer is: {optimizer_type}')
    print(f'learning rate is: {learning_rate}')
    print(f'weight decay is: {weight_decay}')
    
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    return optimizer