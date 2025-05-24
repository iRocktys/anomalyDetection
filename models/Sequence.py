import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, path, sequence_length, column_to_remove=None, normalize=True, mode=None):
        df = pd.read_csv(path, sep=';')

        if column_to_remove and column_to_remove in df.columns:
            df = df.drop(columns=[column_to_remove])

        # data = df.values
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features = scaler.fit_transform(features)

        sequences = []
        sequence_labels = []
        for i in range(len(features) - sequence_length + 1):
            seq = features[i:i+sequence_length]
            label = labels[i+sequence_length-1]  
            sequences.append(seq)
            sequence_labels.append(label)
        
        arr = np.stack(sequences, axis=0).astype(np.float32)   # shape: [n_samples, seq_len, n_features]
        x   = torch.from_numpy(arr)
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
        elif mode == 'cnn2d':
            x = x.unsqueeze(-1)          # [N,5,9,1]
            x = x.permute(0, 2, 1, 3)    # [N, 9, 5, 1]
            self.sequences = x
        else:
            raise ValueError(f"Modo desconhecido '{mode}'. Escolha 'lstm'  ou 'cnn1d'")

        self.labels = y
        self.mode = mode

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
