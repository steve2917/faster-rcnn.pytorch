import torch.nn as nn

class HeadNet(nn.Module):
    def __init__(self, numClasses):
        super(HeadNet, self).__init__()
        self.fc = nn.Linear(1024, numClasses)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x