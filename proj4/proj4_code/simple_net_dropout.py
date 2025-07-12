import torch
import torch.nn as nn

class SimpleNetDropout(nn.Module):
    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
        to understand what it means
        '''
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=16 * 16 * 16, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=15)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        '''
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        y = self.fc_layers(x)

        return y