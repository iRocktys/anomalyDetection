# ===== Cell 1: models/LSTM/ModelLSTM.py =====
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # primeira LSTM: half hidden  
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size//2,
                             num_layers=num_layers,
                             batch_first=True)
        # segunda LSTM: full hidden + dropout  
        self.lstm2 = nn.LSTM(input_size=hidden_size//2,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2)
        # terceira LSTM: back to half hidden + dropout  
        self.lstm3 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size//2,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2)
        # saída  
        self.sigmoid = nn.Sigmoid()
        self.fc      = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        # x: [B, seq_len, input_size]
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out     = self.sigmoid(out)
        return self.fc(out[:, -1, :])  # pega a última saída

    def train_model(self,
                    train_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 100,
                    lr: float = 1e-3,
                    save_path: str = 'output/LSTM/lstm.pth'):
        """Treina a LSTM e salva checkpoint."""
        self.to(device).train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for ep in range(1, epochs+1):
            total_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss   = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            print(f"Epoch {ep}/{epochs} – Loss: {total_loss/n:.4f}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return self

    def evaluate(self,
                 loader: DataLoader,
                 device: str = 'cpu'):
        """Avalia a LSTM no DataLoader fornecido."""
        self.to(device).eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = self(x)
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.extend(preds)
                all_true.extend(y.numpy())
        print("\n=== Classification Report ===")
        print(classification_report(all_true, all_pred))
        print("Accuracy:", accuracy_score(all_true, all_pred))
