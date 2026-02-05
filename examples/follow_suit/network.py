import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # 3 channels; 64 is the output channel size; 3 is the kernel size;
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(17280, 512)
        self.fc2 = nn.Linear(512, 52)

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x, marg_idx=None, type=1):
        if x.dim() == 5:
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.pool(self.ReLU(self.conv3(x)))
        x = self.pool(self.ReLU(self.conv4(x)))

        x = self.flatten(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

