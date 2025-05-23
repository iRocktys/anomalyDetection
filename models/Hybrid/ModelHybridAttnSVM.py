# models/Hybrid/ModelHybridAttnSVM.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelHybridAttnSVM(nn.Module):
    def __init__(self,
                 seq_len: int,
                 n_features: int,
                 lstm_hidden: int,
                 lstm_layers: int,
                 num_classes: int):
        super().__init__()
        # --- CNN2D branch (sai 64 canais) ---
        self.conv1 = nn.Conv2d(1,   32, kernel_size=(3,3), padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=(3,3), padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d((2,1))       # só reduz dimensão temporal
        self.drop  = nn.Dropout(0.3)
        self.adapt = nn.AdaptiveAvgPool2d((1,1))

        # --- LSTM branch ---
        self.lstm = nn.LSTM(input_size = n_features,
                            hidden_size= lstm_hidden,
                            num_layers = lstm_layers,
                            batch_first= True)

        # --- Attention weights (aplica-se em cima de out:[B, seq_len, lstm_hidden]) ---
        self.attn_w = nn.Parameter(torch.randn(seq_len, lstm_hidden))

        # --- Classifier MLP (64 do CNN + lstm_hidden do LSTM) ---
        self.fc1 = nn.Linear(64 + lstm_hidden, 128)   # <-- 64 + lstm_hidden, não 32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, seq_len, n_features]
        B, L, Ff = x.size()

        # CNN2D branch
        c = x.unsqueeze(1)                   # [B,1,L,Ff]
        c = F.relu(self.bn1(self.conv1(c)))
        c = self.pool(c)                     # [B,32,⌊L/2⌋,Ff]
        c = F.relu(self.bn2(self.conv2(c)))
        c = self.pool(c)                     # [B,64,⌊L/4⌋,Ff]
        c = self.drop(c)
        c = self.adapt(c)                    # [B,64,1,1]
        c = c.view(B, 64)                    # [B,64]

        # LSTM branch
        out, (hn, _) = self.lstm(x)          # out:[B,L,lstm_hidden]
        # Attention: score[i] = mean_j out[:,j]·w[j] → softmax → context
        scores = torch.matmul(out, self.attn_w.t())  # [B,L,LstH]→... simplificamos
        α = F.softmax(scores.mean(-1), dim=1)        # [B,L]
        context = torch.bmm(α.unsqueeze(1), out).squeeze(1)  # [B,lstm_hidden]

        # concat & classifier
        f = torch.cat([c, context], dim=1)  # [B, 64 + lstm_hidden]
        f = F.relu(self.fc1(f))
        return self.fc2(f)                  # [B,num_classes]

    def extract_features(self, x):
        """
        Extrai o vetor de features [B,64 + lstm_hidden] antes do fc final.
        """
        B, L, Ff = x.size()
        # CNN2D branch
        c = x.unsqueeze(1)
        c = F.relu(self.bn1(self.conv1(c)))
        c = self.pool(c)
        c = F.relu(self.bn2(self.conv2(c)))
        c = self.pool(c)
        c = self.drop(c)
        c = self.adapt(c)
        c = c.view(B, 64)

        # LSTM branch
        out, (hn, _) = self.lstm(x)
        scores = torch.matmul(out, self.attn_w.t())
        α = F.softmax(scores.mean(-1), dim=1)
        context = torch.bmm(α.unsqueeze(1), out).squeeze(1)

        return torch.cat([c, context], dim=1)  # [B,64 + lstm_hidden]
