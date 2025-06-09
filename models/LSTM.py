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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # 1ª camada LSTM (reduce half hidden)
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size // 2,
                             num_layers=num_layers,
                             batch_first=True)
        # 2ª camada LSTM (full hidden + dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size // 2,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2)
        # 3ª camada LSTM (back to half hidden + dropout)
        self.lstm3 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size // 2,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=0.2)
        # cabeça final
        self.fc = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x: [B, seq_len, input_size]
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        # pegamos somente o último passo temporal
        return self.fc(out[:, -1, :])

    def train_model(self,
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    device: str = 'cpu',
                    epochs: int = 100,
                    lr: float = 1e-3,
                    save_dir: str = 'output/LSTM',
                    threshold: float = 0.87,
                    patience: int = 5):
        """
        Treina a LSTM com EarlyStopping e ReduceLROnPlateau.
        Salva sempre no mesmo arquivo 'LSTM_best_model.pth' quando:
          - val_acc >= threshold
          - E val_acc > best_acc OU val_loss < best_loss
        """
        os.makedirs(save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_loss = float('inf')
        epochs_no_improve = 0

        for ep in range(1, epochs + 1):
            # — treinamento —
            self.train().to(device)
            total_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = total_loss / n

            # — validação —
            self.eval().to(device)
            val_loss, m = 0.0, 0
            preds, trues = [], []
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    out = self(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    m += x.size(0)
                    p = torch.argmax(out, dim=1).cpu().numpy()
                    preds.extend(p)
                    trues.extend(y.cpu().numpy())
            val_loss /= m
            val_acc  = accuracy_score(trues, preds)
            val_f1   = f1_score(trues, preds, average='weighted')

            print(f"Epoch {ep}/{epochs} – "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Val Acc: {val_acc:.4f}  "
                  f"Val F1: {val_f1:.4f}")

            # scheduler ajusta lr pela perda de validação
            scheduler.step(val_loss)

            # critério de salvamento
            improved = (val_acc > best_acc) or (val_loss < best_loss)
            if val_acc >= threshold and improved:
                best_acc = max(val_acc, best_acc)
                best_loss = min(val_loss, best_loss)
                epochs_no_improve = 0
                ckpt_path = os.path.join(save_dir, 'LSTM_best_model.pth')
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
    #     """Gera classification report e accuracy no loader fornecido."""
    #     self.to(device).eval()
    #     all_pred, all_true = [], []
    #     with torch.no_grad():
    #         for x, y in loader:
    #             x = x.to(device)
    #             out = self(x)
    #             preds = torch.argmax(out, dim=1).cpu().numpy()
    #             all_pred.extend(preds)
    #             all_true.extend(y.numpy())

    #     print("\n=== Classification Report ===")
    #     print(classification_report(all_true, all_pred))
    #     print("Accuracy:", accuracy_score(all_true, all_pred))

    
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
      plt.figure()
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
      plt.title('Matriz de Confusão')
      plt.xlabel('Predito')
      plt.ylabel('Real')
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'matriz_confusao.png'))
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
      plt.savefig(os.path.join(result_dir, 'precision_vs_recall.png'))
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
      plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
      plt.close()