import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, inputShape: int, hiddenUnits: int, outputShape: int):
        super().__init__()
        self.vggBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                out_channels=hiddenUnits, kernel_size=3,
                stride=1,padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnits, 
                out_channels=hiddenUnits,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                stride=2
            )
        )
        self.vggBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenUnits,
                out_channels=hiddenUnits,
                kernel_size=3,
                stride=1,padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenUnits, 
                out_channels=hiddenUnits,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenUnits*7*7, 
                out_features=outputShape
            )
        )
    
    def forward(self, x):
        x = self.vggBlock1(x)
        x = self.vggBlock2(x)
        x = self.classifier(x)
        return x