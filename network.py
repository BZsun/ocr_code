import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace

class OcrModel(nn.Module):
    def __init__(self, num_classes):
        super(OcrModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Recurrent layers
        self.rnn = nn.GRU(512, num_classes, bidirectional=True, batch_first=True)

        # Output layers
        self.classifier = nn.Linear(num_classes*2, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(self.conv5(x))
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        
        # Recurrent layers
        x = x.permute(0, 3, 1, 2)  # (batch_size, seq_len, num_channels, height)
        x = x.reshape(x.size(0), x.size(1), -1)  # (batch_size, seq_len, num_channels * height)
        x, _ = self.rnn(x)
        
        # Output layers
        x = self.classifier(x)
        
        return x
