import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class SequenceDataset(Dataset):
    def __init__(self, path, sequence_length, column_to_remove=None, normalize=True, mode='lstm'):
        df = pd.read_csv(path, sep=';')

        if column_to_remove and column_to_remove in df.columns:
            df = df.drop(columns=[column_to_remove])

        data = df.values
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        if normalize:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        sequences = []
        sequence_labels = []
        for i in range(len(features) - sequence_length + 1):
            seq = features[i:i+sequence_length]
            label = labels[i+sequence_length-1]  
            sequences.append(seq)
            sequence_labels.append(label)
        
        # Tensor inicial: [n_samples, seq_len, n_features]
        x = torch.tensor(sequences, dtype=torch.float32)
        y = torch.tensor(sequence_labels, dtype=torch.long)

        # Ajusta formato conforme o modo desejado
        mode = mode.lower()
        if mode == 'lstm':
            # retorna [n_samples, seq_len, n_features]
            self.sequences = x
        elif mode == 'cnn1d':
            # x: [n_samples, seq_len, n_features]
            x = x.permute(0, 2, 1)            
            self.sequences = x
        else:
            raise ValueError(f"Modo desconhecido '{mode}'. Escolha 'lstm', 'cnn1d' ou 'cnn2d'.")

        self.labels = y
        self.mode = mode

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
