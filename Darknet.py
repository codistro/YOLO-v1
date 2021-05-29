import torch
from torch import nn
from torch.nn.modules import padding
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.conv import Conv2d

class Darknet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(192, 128, 1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),

            #4X
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),

            #2X
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),

            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU()

        )

        self.fc = nn.Linear(7 * 7 * 1024, 4096)

        self.detector = nn.Linear(4096, 7 * 7 * 30)

        
    def forward(self, X):
        output = self.net(X)

        output = output.flatten()

        output = self.fc(output)

        output = self.detector(output)

        return output


img = torch.zeros((1, 3, 448, 448))
model = Darknet()
output = model(img)
print(output.shape)
