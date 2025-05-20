import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FlowSequenceDataset(Dataset):
    """
    Dataset para janelas de sequência sobre um CSV já pré-split.
    """
    def __init__(self, csv_path: str, seq_len: int = 10, label_col: str = 'label'):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        # separa features e labels
        features = df.drop(columns=[label_col]).values.astype(np.float32)
        labels = df[label_col].values
        sequences, seq_labels = [], []
        # gera janelas de tamanho seq_len
        for i in range(len(df) - seq_len + 1):
            window = features[i:i+seq_len]
            sequences.append(window)
            window_labels = labels[i:i+seq_len]
            seq_labels.append(np.bincount(window_labels).argmax())
        self.sequences = np.stack(sequences)
        self.seq_labels = np.array(seq_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx]
