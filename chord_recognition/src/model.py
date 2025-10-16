import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNChordNet(nn.Module):
    """
    CNN za hromagrame (B, 1, 12, T). Više nema hard-kodiranog fc ulaza.
    - Dve konvolucije (1->16->32)
    - MaxPool(2x2)
    - AdaptiveAvgPool2d((1,1)) = global average pooling
    - Mali MLP head: 32 -> 64 -> num_classes
    """
    def __init__(self, num_classes=61, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling: (B, 32, H, W) -> (B, 32, 1, 1)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))

        # Head više ne zavisi od H×W — samo od broja kanala (32)
        self.fc1   = nn.Linear(32, 64)
        self.fc2   = nn.Linear(64, num_classes)
        self.do    = nn.Dropout(p=dropout_p)

        # (nije obavezno) lakša inicijalizacija
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, 12, T)
        x = F.relu(self.conv1(x))        # (B,16,12,T)
        x = F.relu(self.conv2(x))        # (B,32,12,T)
        x = self.pool(x)                 # (B,32,6, T/2)  ali nije bitno kolike su dimenzije

        x = self.gap(x)                  # (B,32,1,1)
        x = x.view(x.size(0), -1)        # (B,32)

        x = self.do(F.relu(self.fc1(x))) # (B,64)
        x = self.fc2(x)                  # (B,num_classes)
        return x
