import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelHybrid(nn.Module):
    def __init__(self,
                 seq_len: int,
                 n_features: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int,
                 num_classes: int):
        """
        seq_len          -> comprimento da janela
        n_features       -> features por passo
        lstm_hidden_size -> tamanho do hidden da LSTM
        lstm_num_layers  -> camadas empilhadas da LSTM
        num_classes      -> número de classes finais
        """
        super().__init__()

        # CNN2D branch
        self.conv1 = nn.Conv2d(1,   32, kernel_size=(3,3), padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=(3,3), padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256, kernel_size=(3,3), padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d((2,1))
        self.drop  = nn.Dropout(0.4)
        self.adapt = nn.AdaptiveAvgPool2d((1,1))

        # LSTM branch
        self.lstm = nn.LSTM(input_size   = n_features,
                            hidden_size  = lstm_hidden_size,
                            num_layers   = lstm_num_layers,
                            batch_first  = True)

        # Classifier
        self.fc1 = nn.Linear(256 + lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, seq_len, n_features]
        B = x.size(0)

        # CNN2D branch
        x_c = x.unsqueeze(1)                  # [B,1,seq_len,n_features]
        x_c = F.relu(self.bn1(self.conv1(x_c)))
        x_c = F.relu(self.bn2(self.conv2(x_c)))
        x_c = self.pool(x_c)
        x_c = F.relu(self.bn3(self.conv3(x_c)))
        x_c = F.relu(self.bn4(self.conv4(x_c)))
        x_c = self.pool(x_c)
        x_c = self.drop(x_c)
        x_c = self.adapt(x_c)                 # [B,256,1,1]
        x_c = x_c.view(B, 256)                # [B,256]

        # LSTM branch
        _, (h_n, _) = self.lstm(x)            # h_n[-1]: [B, hidden]
        h_last = h_n[-1]

        # concat & classify
        f = torch.cat([x_c, h_last], dim=1)   # [B,256+hidden]
        f = F.relu(self.fc1(f))
        return self.fc2(f)                    # [B,num_classes]

    def extract_features(self, x):
        """Retorna [B, 256 + hidden_size] sem passar pelo fc final."""
        B = x.size(0)
        # mesma rotina CNN+LSTM que no forward, mas sem a MLP final
        x_c = x.unsqueeze(1)
        x_c = F.relu(self.bn1(self.conv1(x_c)))
        x_c = F.relu(self.bn2(self.conv2(x_c)))
        x_c = self.pool(x_c)
        x_c = F.relu(self.bn3(self.conv3(x_c)))
        x_c = F.relu(self.bn4(self.conv4(x_c)))
        x_c = self.pool(x_c)
        x_c = self.drop(x_c)
        x_c = self.adapt(x_c)
        x_c = x_c.view(B, 256)

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]

        return torch.cat([x_c, h_last], dim=1)
