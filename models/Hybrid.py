import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader

class ModelHybridAttnSVM(nn.Module):
    def __init__(self,
                 seq_len: int,
                 n_features: int,
                 lstm_hidden: int,
                 lstm_layers: int,
                 num_classes: int,
                 pca_components: int = 30,
                 svm_C: float = 1.0):
        super().__init__()
        # --- CNN1D branch: 3 blocos conv->ReLU->Pool ---
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop  = nn.Dropout(0.5)

        # --- LSTM branch sobre saída da CNN ---
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)

        # --- Classifier MLP ---
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # --- PCA + SVM para classificação final ---
        self.pca = PCA(n_components=pca_components)
        self.svm = SVC(kernel='rbf', C=svm_C, probability=True)

    def forward(self, x):
        # x: [B, seq_len, n_features]
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)
        x = x.permute(0, 2, 1)

        out, (h_n, _) = self.lstm(x)
        feats = h_n[-1]  # [B, lstm_hidden]

        f = self.relu(self.fc1(feats))
        return self.fc2(f)

    @torch.no_grad()
    def extract_features(self, x):
        # garante que x e pesos estão no mesmo device
        device = next(self.parameters()).device
        x = x.to(device)
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)
        x = x.permute(0, 2, 1)

        out, (h_n, _) = self.lstm(x)
        feats = h_n[-1]              # [B, lstm_hidden]
        return feats.cpu().numpy()

    def train_model(self, train_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 15,
                    lr: float = 1e-3,
                    save_path: str = 'output/Hybrid/hybrid_attn.pth'):
        self.to(device).train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, epochs+1):
            total_loss, count = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/count:.4f}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return self

    def train_svm(self, train_loader: DataLoader,
                  device: str = 'cpu',
                  pca_path: str = 'output/Hybrid/pca.joblib',
                  svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
        self.to(device).eval()
        feats_list, labels_list = [], []
        for x, y in train_loader:
            f = self.extract_features(x)  # x levado ao device e sem grad
            feats_list.append(f)
            labels_list.append(y.numpy())
        X = np.vstack(feats_list)
        y = np.concatenate(labels_list)

        X_pca = self.pca.fit_transform(X)
        self.svm.fit(X_pca, y)
        joblib.dump(self.pca, pca_path)
        joblib.dump(self.svm, svm_path)
        print(f"PCA salvo em {pca_path}\nSVM salvo em {svm_path}")
        return self.svm

    def evaluate(self, loader: DataLoader,
                 device: str = 'cpu',
                 pca_path: str = 'output/Hybrid/pca.joblib',
                 svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
        pca = self.pca if hasattr(self, 'pca') else joblib.load(pca_path)
        svm = self.svm if hasattr(self, 'svm') else joblib.load(svm_path)
        feats_list, labels_list = [], []
        for x, y in loader:
            f = self.extract_features(x)
            feats_list.append(f)
            labels_list.append(y.numpy())
        X = np.vstack(feats_list)
        y_true = np.concatenate(labels_list)
        X_pca = pca.transform(X)
        y_pred = svm.predict(X_pca)
        print(classification_report(y_true, y_pred))
        print("Accuracy:", accuracy_score(y_true, y_pred))
