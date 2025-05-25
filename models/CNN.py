import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score,classification_report

class CNN(nn.Module):
    def __init__(self, input_channels, input_length, num_classes):
        super().__init__()
        # Blocos convolucionais
        self.conv1   = nn.Conv1d(input_channels,  32, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm1d(32)
        self.conv2   = nn.Conv1d(32,  64, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm1d(64)
        self.conv3   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3     = nn.BatchNorm1d(128)
        self.conv4   = nn.Conv1d(128,256, kernel_size=3, padding=1)
        self.bn4     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)

        # Fully connected
        self.fc1 = nn.Linear(256 * input_length, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, n_features, seq_len]
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
                    threshold: float = 0.87,
                    patience: int = 5):
        """
        Treina a CNN com EarlyStopping e ReduceLROnPlateau.
        Salva sempre no mesmo arquivo 'CNN_best_model.pth' quando:
          - val_acc >= threshold
          - AND (val_acc > best_acc OR val_loss < best_loss)
        """
        os.makedirs(save_dir, exist_ok=True)
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_loss = float('inf')
        epochs_no_improve = 0

        for ep in range(1, epochs + 1):
            # --- Treinamento ---
            self.train()
            train_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out  = self(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                n += x.size(0)
            train_loss /= n

            # --- Validação ---
            self.eval()
            val_loss, m = 0.0, 0
            preds, trues = [], []
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    out  = self(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    m += x.size(0)
                    p = torch.argmax(out, dim=1).cpu().numpy()
                    preds.extend(p)
                    trues.extend(y.cpu().numpy())
            val_loss /= m
            val_acc  = accuracy_score(trues, preds)

            print(f"Epoch {ep}/{epochs} – "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Val Loss:   {val_loss:.4f}  "
                  f"Val Acc:    {val_acc:.4f}")

            # Ajusta taxa de aprendizado
            scheduler.step(val_loss)

            # Critério de salvamento
            improved = (val_acc > best_acc) or (val_loss < best_loss)
            if val_acc >= threshold and improved:
                best_acc = max(val_acc, best_acc)
                best_loss = min(val_loss, best_loss)
                epochs_no_improve = 0
                ckpt_path = os.path.join(save_dir, 'CNN_best_model.pth')
                torch.save(self.state_dict(), ckpt_path)
                print(f"→ Novo melhor modelo salvo em '{ckpt_path}'")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping após {ep} épocas sem melhora.")
                    break

        return self

    def evaluate(self,
                 loader: DataLoader,
                 device: str = 'cpu'):
        """Avalia a CNN no DataLoader fornecido."""
        self.to(device).eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = self(x)
                p   = torch.argmax(out, dim=1).cpu().numpy()
                preds.extend(p)
                trues.extend(y.cpu().numpy())

        print("\n=== Classification Report ===")
        print(classification_report(trues, preds))
        print("Accuracy:", accuracy_score(trues, preds))
