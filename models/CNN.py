import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self, input_channels, input_length, num_classes):
        super().__init__()
        # Bloco conv1
        self.conv1 = nn.Conv1d(input_channels,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        # Bloco conv2
        self.conv2 = nn.Conv1d(32,  64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        # Bloco conv3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        # Bloco conv4
        self.conv4 = nn.Conv1d(128,256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)

        # Fully connected
        self.fc1 = nn.Linear(256 * input_length, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x já é [B, n_features, seq_len], então NÃO permutar
        x = F.relu(self.bn1(self.conv1(x)))   # [B,32, seq_len]
        x = F.relu(self.bn2(self.conv2(x)))   # [B,64, seq_len]
        x = F.relu(self.bn3(self.conv3(x)))   # [B,128, seq_len]
        x = F.relu(self.bn4(self.conv4(x)))   # [B,256, seq_len]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)             # [B, 256*seq_len]
        x = F.relu(self.fc1(x))               # [B, 128]
        return self.fc2(x)                    # [B, num_classes]

    def train_model(self,
                    train_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 20,
                    lr: float = 1e-3,
                    save_path: str = 'output/CNN/cnn_model.pth'):
        """Treina a CNN e salva os pesos."""
        self.to(device).train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs+1):
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
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/n:.4f}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return self

    def evaluate(self,
                 loader: DataLoader,
                 device: str = 'cpu'):
        """Avalia a CNN no DataLoader fornecido."""
        self.to(device).eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = self(x)
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        print("\n=== Classification Report ===")
        print(classification_report(all_labels, all_preds))
        print("Accuracy:", accuracy_score(all_labels, all_preds))
