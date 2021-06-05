from torch import nn
from torchvision import models


class Darknet(nn.Module):

    def __init__(self, S=7, B=2, classes=80):
        super().__init__()

        self.net = models.resnet50()

        self.net.fc = nn.Linear(2048, 2048)

        self.detector = nn.Linear(2048, S * S * (5 * B + classes))

    def forward(self, X):
        output = self.net(X)

        output = self.detector(output)

        return output
