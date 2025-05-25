import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # 1ª camada LSTM (reduce half hidden)
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size // 2,
                             num_layers=num_layers,
                             batch_first=True)
        # 2ª camada LSTM (full hidden + dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size // 2,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2)
        # 3ª camada LSTM (back to half hidden + dropout)
        self.lstm3 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size // 2,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2)
        # cabeça final
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x: [B, seq_len, input_size]
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        # pegamos somente o último passo temporal
        return self.fc(out[:, -1, :])

    def train_model(self,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 100,
                    lr: float = 1e-3,
                    save_dir: str = 'output/LSTM',
                    threshold: float = 0.87):
        """
        Treina a LSTM e só salva checkpoints quando a acurácia
        de validação ultrapassar `threshold`. Nome do arquivo:
        LSTM_<accuracy>.pth
        """
        os.makedirs(save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        for ep in range(1, epochs + 1):
            # garante modo train antes de cada época
            self.to(device).train()
            total_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = total_loss / n

            # avaliação
            self.to(device).eval()
            all_pred, all_true = [], []
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    out = self(x)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    all_pred.extend(preds)
                    all_true.extend(y.numpy())
            val_acc = accuracy_score(all_true, all_pred)

            print(f"Ep {ep}/{epochs} — Train Loss: {train_loss:.4f} — Val Acc: {val_acc:.4f}")

            # salva apenas se superar threshold e for melhor que best_acc
            if val_acc >= threshold and val_acc > best_acc:
                best_acc = val_acc
                fname = f"LSTM_{val_acc:.2f}.pth"
                torch.save(self.state_dict(), os.path.join(save_dir, fname))
                print(f"→ Checkpoint salvo: {fname}")

        return self

    def evaluate(self,
                 loader: DataLoader,
                 device: str = 'cpu'):
        """Gera classification report e accuracy no loader fornecido."""
        self.to(device).eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                out = self(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_pred.extend(preds)
                all_true.extend(y.numpy())

        print("\n=== Classification Report ===")
        print(classification_report(all_true, all_pred))
        print("Accuracy:", accuracy_score(all_true, all_pred))
