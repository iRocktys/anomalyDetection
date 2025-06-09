import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
)


class CNN(nn.Module):
    # MODELO 1 - Inicial (modelo 2 foi considerado melhor pelo grafico de treinamento do Loss na validação e treinamento)
    # def __init__(self, input_channels, input_length, num_classes):
    #     super().__init__()
    #     # Blocos convolucionais
    #     self.conv1   = nn.Conv1d(input_channels,  32, kernel_size=3, padding=1)
    #     self.bn1     = nn.BatchNorm1d(32)
    #     self.conv2   = nn.Conv1d(32,  64, kernel_size=3, padding=1)
    #     self.bn2     = nn.BatchNorm1d(64)
    #     self.conv3   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
    #     self.bn3     = nn.BatchNorm1d(128)
    #     self.conv4   = nn.Conv1d(128,256, kernel_size=3, padding=1)
    #     self.bn4     = nn.BatchNorm1d(256)
    #     self.dropout = nn.Dropout(p=0.4)

    #     # Fully connected
    #     self.fc1 = nn.Linear(256 * input_length, 128)
    #     self.fc2 = nn.Linear(128, num_classes)

    # def forward(self, x):
    #     # x: [B, n_features, seq_len]
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = F.relu(self.bn4(self.conv4(x)))
    #     x = self.dropout(x)
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc1(x))
    #     return self.fc2(x)
    
    # MODELO 2 - pooling e GAP - MELHOR ATÉ ENTÃO
    def __init__(self, input_channels, input_length, num_classes):
      super().__init__()

      self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
      self.pool1 = nn.MaxPool1d(kernel_size=2)

      self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
      self.pool2 = nn.MaxPool1d(kernel_size=2)

      self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
      self.pool3 = nn.MaxPool1d(kernel_size=2)

      self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
      self.pool4 = nn.MaxPool1d(kernel_size=2)

      self.dropout = nn.Dropout(p=0.4)

      self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
      self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
      x = self.pool1(F.relu(self.conv1(x)))
      x = self.pool2(F.relu(self.conv2(x)))
      x = self.pool3(F.relu(self.conv3(x)))
      x = self.pool4(F.relu(self.conv4(x)))
      x = self.dropout(x)
      x = self.global_avg_pool(x)  # [B, 256, 1]
      x = x.squeeze(-1)            # [B, 256]
      return self.fc(x)            # [B, num_classes]

    # MODELO 3 - pooling, GAP e BatchNorm1d
    # def __init__(self, input_channels, input_length, num_classes):
    #   super().__init__()

    #   self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
    #   self.bn1   = nn.BatchNorm1d(32)
    #   self.pool1 = nn.MaxPool1d(kernel_size=2)

    #   self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
    #   self.bn2   = nn.BatchNorm1d(64)
    #   self.pool2 = nn.MaxPool1d(kernel_size=2)

    #   self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
    #   self.bn3   = nn.BatchNorm1d(128)
    #   self.pool3 = nn.MaxPool1d(kernel_size=2)

    #   self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
    #   self.bn4   = nn.BatchNorm1d(256)
    #   self.pool4 = nn.MaxPool1d(kernel_size=2)

    #   self.dropout = nn.Dropout(p=0.4)

    #   # Agrupamento médio global (GAP) substitui FC
    #   self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    #   self.fc = nn.Linear(256, num_classes)

    # def forward(self, x):
    #   x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    #   x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    #   x = self.pool3(F.relu(self.bn3(self.conv3(x))))
    #   x = self.pool4(F.relu(self.bn4(self.conv4(x))))
    #   x = self.dropout(x)
    #   x = self.global_avg_pool(x)  # shape: [B, 256, 1]
    #   x = x.squeeze(-1)            # shape: [B, 256]
    #   return self.fc(x)            # shape: [B, num_classes]


    def train_model(self,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 20,
                    lr: float = 1e-3,
                    save_dir: str = 'output/CNN',
                    threshold: float = 0.87,
                    patience: int = 20):
      
        os.makedirs(save_dir, exist_ok=True)
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_loss = float('inf')
        epochs_no_improve = 0

        for ep in range(1, epochs + 1):
            # --- Treinamento ---
            self.train()
            train_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out  = self(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                n += x.size(0)
            train_loss /= n

            # --- Validação ---
            self.eval()
            val_loss, m = 0.0, 0
            preds, trues = [], []
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    out  = self(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    m += x.size(0)
                    p = torch.argmax(out, dim=1).cpu().numpy()
                    preds.extend(p)
                    trues.extend(y.cpu().numpy())
            val_loss /= m
            val_acc  = accuracy_score(trues, preds)

            print(f"Epoch {ep}/{epochs} – "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Val Loss:   {val_loss:.4f}  "
                  f"Val Acc:    {val_acc:.4f}")

            # Ajusta taxa de aprendizado
            scheduler.step(val_loss)

            # Critério de salvamento
            improved = (val_acc > best_acc) or (val_loss < best_loss)
            if val_acc >= threshold and improved:
                best_acc = max(val_acc, best_acc)
                best_loss = min(val_loss, best_loss)
                epochs_no_improve = 0
                ckpt_path = os.path.join(save_dir, 'CNN_best_model.pth')
                torch.save(self.state_dict(), ckpt_path)
                print(f"→ Novo melhor modelo salvo em '{ckpt_path}'")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping após {ep} épocas sem melhora.")
                    break

        return self

    # def evaluate(self,
    #              loader: DataLoader,
    #              device: str = 'cpu'):
    #     """Avalia a CNN no DataLoader fornecido."""
    #     self.to(device).eval()
    #     preds, trues = [], []
    #     with torch.no_grad():
    #         for x, y in loader:
    #             x, y = x.to(device), y.to(device)
    #             out = self(x)
    #             p   = torch.argmax(out, dim=1).cpu().numpy()
    #             preds.extend(p)
    #             trues.extend(y.cpu().numpy())

    #     print("\n=== Classification Report ===")
    #     print(classification_report(trues, preds))
    #     print("Accuracy:", accuracy_score(trues, preds))
    
    def evaluate(self, loader: DataLoader, device: str = 'cpu'):
      """Avalia o modelo binário e gera gráficos e métricas salvas em arquivos."""
      self.to(device).eval()
      preds, trues, probs = [], [], []

      with torch.no_grad():
          for x, y in loader:
              x, y = x.to(device), y.to(device)
              out = self(x)
              prob = torch.softmax(out, dim=1).cpu().numpy()

              if prob.shape[1] == 2:
                  probs.extend(prob[:, 1])  # probabilidade da classe positiva
              else:
                  raise ValueError(
                      f"Esperado prob.shape[1] == 2 para classificação binária com CrossEntropyLoss, mas recebido shape {prob.shape}"
                  )

              preds.extend(np.argmax(prob, axis=1))
              trues.extend(y.cpu().numpy())

      trues = np.array(trues)
      preds = np.array(preds)
      probs = np.array(probs)

      result_dir = os.path.join('output', 'resultados', self.__class__.__name__)
      os.makedirs(result_dir, exist_ok=True)

      # Métricas principais
      acc     = accuracy_score(trues, preds)
      prec    = precision_score(trues, preds)
      rec     = recall_score(trues, preds)
      f1      = f1_score(trues, preds)
      fpr, tpr, _ = roc_curve(trues, probs)
      roc_auc = auc(fpr, tpr)
      report  = classification_report(trues, preds, digits=4)

      # Exibir no console
      print("\n=== Classification Report ===")
      print(report)
      print(f"Acurácia:  {acc:.4f}")
      print(f"Precisão:  {prec:.4f}")
      print(f"Recall:    {rec:.4f}")
      print(f"F1-Score:  {f1:.4f}")
      print(f"AUC:       {roc_auc:.4f}")

      # Salvar em classification_report.txt
      with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
          f.write("=== Classification Report ===\n")
          f.write(report)
          f.write("\n\n")
          f.write(f"Acurácia:       {acc:.4f}\n")
          f.write(f"Precisão:       {prec:.4f}\n")
          f.write(f"Recall:         {rec:.4f}\n")
          f.write(f"F1-Score:       {f1:.4f}\n")
          f.write(f"AUC:            {roc_auc:.4f}\n")

      # Salvar em metrics.json
      metrics_dict = {
          "acuracia":  round(acc, 4),
          "precisao":  round(prec, 4),
          "recall":    round(rec, 4),
          "f1_score":  round(f1, 4),
          "auc":       round(roc_auc, 4)
      }
      with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
          json.dump(metrics_dict, f, indent=4)

      # Matriz de confusão
      cm = confusion_matrix(trues, preds)
      plt.figure(figsize=(8, 6))  # ← aumenta o tamanho da figura
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 13})  # ← aumenta tamanho dos números
      plt.title('Matriz de Confusão', fontsize=14)
      plt.xlabel('Predito', fontsize=12)
      plt.ylabel('Real', fontsize=12)
      plt.xticks(fontsize=11)
      plt.yticks(fontsize=11)
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'CNN_matriz_confusao.png'))
      plt.savefig(os.path.join(result_dir, 'CNN_matriz_confusao.pdf'))
      plt.close()

      # Precision vs Recall
      precision, recall, _ = precision_recall_curve(trues, probs)
      plt.figure()
      plt.plot(recall, precision, marker='.')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('Precision vs Recall')
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'CNN_precision_vs_recall.pdf'))
      plt.close()

      # ROC Curve
      plt.figure()
      plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
      plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('ROC Curve')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'CNN_roc_curve.pdf'))
      plt.close()

