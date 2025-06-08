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
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(p=0.5)

        # --- Enhanced LSTM branch ---
        self.lstm1 = nn.LSTM(input_size=128,
                             hidden_size=lstm_hidden // 2,
                             num_layers=1,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden // 2,
                             hidden_size=lstm_hidden,
                             num_layers=1,
                             batch_first=True,
                             dropout=0.2)
        self.lstm3 = nn.LSTM(input_size=lstm_hidden,
                             hidden_size=lstm_hidden // 2,
                             num_layers=1,
                             batch_first=True,
                             dropout=0.2)

        # --- MLP head ---
        self.fc1 = nn.Linear(lstm_hidden // 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, n_features]
        x = x.permute(0, 2, 1)  # [B, n_features, seq_len]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)
        x = x.permute(0, 2, 1)  # [B, seq_len//8, 128]

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        feats = x[:, -1, :]  # pega último passo temporal

        f = self.relu(self.fc1(feats))
        return self.fc2(f)

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

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        feats = x[:, -1, :]
        return feats.cpu().numpy()

    def train_model(self,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                device: str = 'cpu',
                epochs: int = 30,
                lr: float = 1e-3,
                save_dir: str = 'output/Hybrid',
                threshold: float = 0.87,
                patience: int = 15):
      """
      Treina CNN+LSTM com Early Stopping + ReduceLROnPlateau,
      salvando checkpoints somente quando:
        - val_acc >= threshold
        - val_acc > best_acc OR val_loss < best_loss
      e nomeando o arquivo como Hybrid_<val_acc>.pth
      """
      os.makedirs(save_dir, exist_ok=True)
      optimizer = torch.optim.Adam(self.parameters(), lr=lr)
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.5,
                                                            patience=20)
      criterion = nn.CrossEntropyLoss()

      best_acc = threshold
      best_loss = float('inf')
      epochs_no_improve = 0

      for ep in range(1, epochs+1):
          # --- Treino ---
          self.train().to(device)
          train_loss, train_n = 0.0, 0
          for x,y in train_loader:
              x, y = x.to(device), y.to(device)
              optimizer.zero_grad()
              logits = self(x)
              loss   = criterion(logits, y)
              loss.backward()
              optimizer.step()
              train_loss += loss.item() * x.size(0)
              train_n    += x.size(0)
          train_loss /= train_n

          # --- Validação ---
          self.eval().to(device)
          val_loss, val_n = 0.0, 0
          preds, trues = [], []
          with torch.no_grad():
              for x,y in valid_loader:
                  x, y = x.to(device), y.to(device)
                  out = self(x)
                  loss = criterion(out, y)
                  val_loss += loss.item() * x.size(0)
                  val_n    += x.size(0)
                  p = torch.argmax(out, dim=1).cpu().numpy()
                  preds.extend(p)
                  trues.extend(y.cpu().numpy())
          val_loss /= val_n
          val_acc  = accuracy_score(trues, preds)

          print(f"Epoch {ep}/{epochs} – "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}")

          # Ajusta lr se necessário
          scheduler.step(val_loss)

          # Critério de salvamento
          improved = (val_acc > best_acc) or (val_loss < best_loss)
          if val_acc >= threshold and improved:
              best_acc = max(val_acc, best_acc)
              best_loss = min(val_loss, best_loss)
              epochs_no_improve = 0
              fname = f"Hybrid_best_model.pth"
              torch.save(self.state_dict(), os.path.join(save_dir, fname))
              print(f"→ Checkpoint salvo: {fname}")
          else:
              epochs_no_improve += 1
              if epochs_no_improve >= patience:
                  print(f"Early stopping após {ep} épocas sem melhora.")
                  break

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
