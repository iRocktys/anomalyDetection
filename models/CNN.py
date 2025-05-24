import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

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
        self.dropout = nn.Dropout(p=0.4)

        # Fully connected
        self.fc1 = nn.Linear(256 * input_length, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x já é [B, n_features, seq_len], sem permutar
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_model(self,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 20,
                    lr: float = 1e-3,
                    save_dir: str = 'output/CNN',
                    threshold: float = 0.87):
        """
        Treina a CNN e salva apenas quando a acurácia de validação
        superar `threshold`. O arquivo será nomeado CNN_<acc>.pth.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        for ep in range(1, epochs+1):
            # modo treinamento
            self.train()
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
            train_loss = total_loss / n

            # modo avaliação
            self.eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    out = self(x)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    all_pred.extend(preds)
                    all_true.extend(y.numpy())
            val_acc = accuracy_score(all_true, all_pred)

            print(f"Ep {ep}/{epochs} – Train Loss: {train_loss:.4f} – Val Acc: {val_acc:.4f}")

            # salva se for melhor e acima do threshold
            if val_acc >= threshold and val_acc > best_acc:
                best_acc = val_acc
                fname = f"CNN_{val_acc:.4f}.pth"
                torch.save(self.state_dict(), os.path.join(save_dir, fname))
                print(f"→ Checkpoint salvo: {fname}")

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
                out = self(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        print("\n=== Classification Report ===")
        print(classification_report(all_labels, all_preds))
        print("Accuracy:", accuracy_score(all_labels, all_preds))
