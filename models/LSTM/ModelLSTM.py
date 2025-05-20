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

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Primeira camada LSTM
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size//2,
                             num_layers=num_layers,
                             batch_first=True
                             
                             )
        
        # Segunda camada LSTM
        self.lstm2 = nn.LSTM(input_size=hidden_size//2,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,  
                             dropout=0.2  # Dropout entre as camadas LSTM                           
                            )
        
        self.lstm3 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size//2,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2  # Dropout entre as camadas LSTM
                             )
        
        # Camada fully-connected
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()  # Para binário
        # self.fc = torch.nn.Linear(hidden_size, 1) # nn.BCEWithLogitsLoss() / nn.BCELoss()
        self.fc = torch.nn.Linear(hidden_size//2, output_size) # nn.CrossEntropyLoss()
        # Camada de ativação softmax
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)        
        out = torch.sigmoid(out)  # Aplicar sigmoid para obter probabilidades / Remover para nn.BCEWithLogitsLoss()
        # out = self.softmax(out)  # Aplicar softmax para obter probabilidades / Remover para nn.BCEWithLogitsLoss()
        out = self.fc(out[:, -1, :])  # Usar a última saída do LSTM como entrada para fc
        
        return out    