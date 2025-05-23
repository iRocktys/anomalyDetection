import torch.nn as nn
import torch.nn.functional as F

class CNN_2D(nn.Module):
    def __init__(self, input_channels, input_length, num_classes):
        """
        input_channels -> número de canais de entrada (ex: 1)
        input_length   -> seq_len legado (não usado diretamente em pooling)
        num_classes    -> número de classes (ataque vs normal)
        """
        super().__init__()
        # 4 blocos Conv2d + BatchNorm2d
        self.conv1 = nn.Conv2d(input_channels,  32, kernel_size=(3,3), padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,  64, kernel_size=(3,3), padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # Pooling apenas na dimensão temporal (altura), mantendo largura ≥1
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        self.dropout = nn.Dropout(p=0.4)
        # Garante saída [batch, 256, 1, 1] de qualquer tamanho espacial ≥(2,1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Full–connected
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x esperado: [batch, input_channels, seq_len, n_features]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)   # reduz seq_len → floor(seq_len/2), mantém n_features

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)   # reduz novamente só seq_len

        x = self.dropout(x)
        x = self.adaptive_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]

        x = F.relu(self.fc1(x))
        return self.fc2(x)
