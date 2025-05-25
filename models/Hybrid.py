import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

class ModelHybridAttnSVM(nn.Module):
    def __init__(self,
                 seq_len: int,
                 n_features: int,
                 lstm_hidden: int,
                 lstm_layers: int,
                 num_classes: int):
        super().__init__()
        # --- CNN1D branch ---
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32,       out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64,       out_channels=128, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop  = nn.Dropout(p=0.5)

        # --- LSTM branch ---
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)

        # --- MLP head ---
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, n_features]
        x = x.permute(0, 2, 1)                   # [B, n_features, seq_len]
        x = self.pool(self.relu(self.conv1(x)))  # [B, 32, seq_len/2]
        x = self.pool(self.relu(self.conv2(x)))  # [B, 64, seq_len/4]
        x = self.pool(self.relu(self.conv3(x)))  # [B, 128, seq_len/8]
        x = self.drop(x)
        x = x.permute(0, 2, 1)                   # [B, seq_len/8, 128]

        out, (h_n, _) = self.lstm(x)
        feats = h_n[-1]                          # [B, lstm_hidden]

        f = self.relu(self.fc1(feats))           # [B, 128]
        return self.fc2(f)                       # [B, num_classes]

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        device = next(self.parameters()).device
        x = x.to(device)
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)
        x = x.permute(0, 2, 1)

        out, (h_n, _) = self.lstm(x)
        return h_n[-1].cpu().numpy()            # [B, lstm_hidden]

    def train_model(self,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 15,
                    lr: float = 1e-3,
                    save_dir: str = 'output/Hybrid',
                    threshold: float = 0.87):
        """
        Treina CNN+LSTM e salva checkpoints somente se val_acc >= threshold.
        Arquivos salvos como ModelHybridAttnSVM_<acc>.pth
        """
        os.makedirs(save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = float('inf')

        for ep in range(1, epochs + 1):
            # --- treinamento ---
            self.to(device).train()
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

            # --- validação ---
            self.to(device).eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    out = self(x)
                    p   = torch.argmax(out, dim=1).cpu().numpy()
                    preds.extend(p)
                    trues.extend(y.cpu().numpy())
            val_acc = accuracy_score(trues, preds)

            print(f"Epoch {ep}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

            # salva se for melhor e acima do threshold
            if val_acc >= threshold and train_loss < best_loss:
                fname = f"Hybrid_{val_acc:.2f}.pth"
                torch.save(self.state_dict(), os.path.join(save_dir, fname))
                print(f"→ Checkpoint salvo: {fname}")

        return self

    def train_svm(self,
                  train_loader: DataLoader,
                  pca_components: int = 30,
                  svm_C: float = 1.0,
                  device: str = 'cpu',
                  pca_path: str = 'output/Hybrid/pca.joblib',
                  svm_path: str = 'output/Hybrid/hybrid_svm.joblib') -> SVC:
        """
        Extrai features, aplica PCA e treina SVM. Retorna o modelo SVM.
        """
        self.to(device).eval()
        feats, trues = [], []
        for x, y in train_loader:
            feat = self.extract_features(x)
            feats.append(feat)
            trues.append(y.cpu().numpy())
        X = np.vstack(feats)
        y = np.concatenate(trues)

        pca = PCA(n_components=pca_components)
        X_pca = pca.fit_transform(X)
        svm = SVC(kernel='rbf', C=svm_C, probability=True)
        svm.fit(X_pca, y)

        joblib.dump(pca, pca_path)
        joblib.dump(svm, svm_path)
        return svm

    def evaluate(self,
                 loader: DataLoader,
                 device: str = 'cpu',
                 pca_path: str = 'output/Hybrid/pca.joblib',
                 svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
        """
        Avalia o pipeline CNN→LSTM→PCA→SVM usando artefatos salvos.
        """
        pca = joblib.load(pca_path)
        svm = joblib.load(svm_path)
        feats, trues = [], []
        for x, y in loader:
            feat = self.extract_features(x)
            feats.append(feat)
            trues.append(y.cpu().numpy())
        X = np.vstack(feats)
        y_true = np.concatenate(trues)
        X_pca  = pca.transform(X)
        y_pred = svm.predict(X_pca)

        print(classification_report(y_true, y_pred))
        print("Accuracy:", accuracy_score(y_true, y_pred))
