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
                 num_classes: int,
                 pca_components: int = 30,
                 svm_C: float = 1.0):
        super().__init__()
        # CNN1D branch
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.5)

        # LSTM branch
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)

        # MLP head
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # PCA + SVM
        self.pca = PCA(n_components=pca_components)
        self.svm = SVC(kernel='rbf', C=svm_C, probability=True)

    def forward(self, x):
        # x: [B, seq_len, n_features]
        x = x.permute(0,2,1)                  # [B,n_feat,seq_len]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)
        x = x.permute(0,2,1)                  # [B,seq_len',128]

        out,(h_n,_) = self.lstm(x)            # h_n[-1]: [B,lstm_hidden]
        feats = h_n[-1]
        f = self.relu(self.fc1(feats))
        return self.fc2(f)

    @torch.no_grad()
    def extract_features(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = x.permute(0,2,1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.drop(x)
        x = x.permute(0,2,1)
        out,(h_n,_) = self.lstm(x)
        return h_n[-1].cpu().numpy()

    def train_model(self,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 15,
                    lr: float = 1e-3,
                    save_dir: str = 'output/Hybrid',
                    threshold: float = 0.87):
        """
        Treina CNN+LSTM e salva checkpoint quando val_acc >= threshold.
        Checkpoint nomeado ModelHybridAttnSVM_<acc>.pth
        """
        os.makedirs(save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        for ep in range(1, epochs+1):
            # treinamento
            self.to(device).train()
            total_loss, n = 0.0, 0
            for x,y in train_loader:
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss   = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*x.size(0)
                n += x.size(0)
            train_loss = total_loss/n

            # validação
            self.to(device).eval()
            preds, trues = [], []
            with torch.no_grad():
                for x,y in valid_loader:
                    x = x.to(device)
                    out = self(x)
                    p = torch.argmax(out,1).cpu().numpy()
                    preds.extend(p); trues.extend(y.numpy())
            val_acc = accuracy_score(trues, preds)

            print(f"Ep {ep}/{epochs} – Train Loss: {train_loss:.4f} – Val Acc: {val_acc:.4f}")

            # salvar melhor
            if val_acc >= threshold and val_acc > best_acc:
                best_acc = val_acc
                fname = f"Hybrid_{val_acc:.2f}.pth"
                torch.save(self.state_dict(), os.path.join(save_dir, fname))
                print(f"→ Checkpoint salvo: {fname}")

        return self

    def train_svm(self,
                  train_loader: DataLoader,
                  device: str = 'cpu',
                  pca_path: str = 'output/Hybrid/pca.joblib',
                  svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
        self.to(device).eval()
        feats, trues = [], []
        for x,y in train_loader:
            f = self.extract_features(x)
            feats.append(f); trues.append(y.numpy())
        X = np.vstack(feats); y = np.concatenate(trues)
        X_pca = self.pca.fit_transform(X)
        self.svm.fit(X_pca, y)
        joblib.dump(self.pca, pca_path)
        joblib.dump(self.svm, svm_path)
        return self.svm

    def evaluate(self,
                 loader: DataLoader,
                 device: str = 'cpu',
                 pca_path: str = 'output/Hybrid/pca.joblib',
                 svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
        """
        Avalia o pipeline CNN→LSTM→PCA→SVM a partir dos artefatos em disco.
        """
        # Carrega sempre dos .joblib
        pca = joblib.load(pca_path)
        svm = joblib.load(svm_path)

        feats_list, labels_list = [], []
        for x, y in loader:
            f = self.extract_features(x)  # já move x ao device
            feats_list.append(f)
            labels_list.append(y.numpy())

        X      = np.vstack(feats_list)
        y_true = np.concatenate(labels_list)

        X_pca  = pca.transform(X)
        y_pred = svm.predict(X_pca)

        print(classification_report(y_true, y_pred))
        print("Accuracy:", accuracy_score(y_true, y_pred))

