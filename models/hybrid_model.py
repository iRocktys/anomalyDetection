import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from models.Hybrid.sequence_hybrid import FlowSequenceDataset

class CNNLSTMExtractor(nn.Module):
    """
    Extrator de features CNN -> LSTM.
    """
    def __init__(self, input_features: int, cnn_channels: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)
        self.hidden_size = lstm_hidden

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0,2,1)               # [batch, features, seq_len]
        x = self.cnn(x)                   # [batch, channels, seq_len//2]
        x = x.permute(0,2,1)             # [batch, seq_len//2, channels]
        _, (h_n, _) = self.lstm(x)       # h_n: [1, batch, hidden]
        return h_n[-1]                   # [batch, hidden]


def train_feature_extractor(train_csv: str,
                            seq_len: int = 10,
                            batch_size: int = 64,
                            epochs: int = 10,
                            lr: float = 1e-3,
                            device: str = 'cpu',
                            save_dir: str = 'models'):
    """
    Treina CNN+LSTM e salva pesos.
    Retorna o extrator treinado e o DataLoader de treino.
    """
    os.makedirs(save_dir, exist_ok=True)
    ds = FlowSequenceDataset(train_csv, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = CNNLSTMExtractor(input_features=ds.sequences.shape[2]).to(device)
    classifier = nn.Linear(model.hidden_size, len(np.unique(ds.seq_labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model(x)
            logits = classifier(feats)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(ds):.4f}")

    torch.save({'extractor': model.state_dict(),
                'classifier': classifier.state_dict()},
               os.path.join(save_dir, 'cnn_lstm.pth'))
    return model, loader


def extract_features(model: nn.Module,
                     loader: DataLoader,
                     device: str = 'cpu'):
    """
    Extrai features usando o extractor CNN-LSTM.
    """
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats.append(model(x).cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)


def train_svm(X: np.ndarray,
              y: np.ndarray,
              kernel: str = 'rbf',
              C: float = 1.0,
              svm_path: str = 'models/svm_model.joblib'):
    """
    Treina e salva um classificador SVM sobre as features.
    """
    svm = SVC(kernel=kernel, C=C, probability=True)
    svm.fit(X, y)
    joblib.dump(svm, svm_path)
    print(f"SVM trained and saved to {svm_path}")
    return svm


def evaluate_hybrid(model: nn.Module,
                    loader: DataLoader,
                    svm: SVC,
                    device: str = 'cpu'):
    """
    Avalia o modelo híbrido no conjunto fornecido.
    """
    feats, trues = extract_features(model, loader, device)
    preds = svm.predict(feats)
    print(classification_report(trues, preds))
    print("Accuracy:", accuracy_score(trues, preds))