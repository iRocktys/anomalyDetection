import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report
)
import csv

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
                             num_layers=3,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden // 2,
                             hidden_size=lstm_hidden,
                             num_layers=3,
                             batch_first=True,
                             dropout=0.2)
        self.lstm3 = nn.LSTM(input_size=lstm_hidden,
                             hidden_size=lstm_hidden // 2,
                             num_layers=3,
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
                patience: int = 10):
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
                                                            patience=10)
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
          precision = precision_score(trues, preds, zero_division=0)
          recall = recall_score(trues, preds, zero_division=0)
          f1 = f1_score(trues, preds, zero_division=0)


          print(f"Epoch {ep}/{epochs} – "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}")
          
          # CSV por época - log simples do treinamento
          model_name = os.path.basename(save_dir.rstrip("/\\"))
          epoch_log_path = os.path.join(save_dir, f"{model_name}_epoch_metrics.csv")
          file_exists = os.path.isfile(epoch_log_path)
          with open(epoch_log_path, mode='a', newline='') as f:
              writer = csv.writer(f)
              if not file_exists:
                  writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'precision', 'recall', 'f1_score'])
              writer.writerow([ep, train_loss, val_loss, val_acc, precision, recall, f1])


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
      Extrai features, aplica PCA e treina SVM. Salva artefatos e métricas.
      """
      self.to(device).eval()
      feats, trues = [], []
      for x, y in train_loader:
          feat = self.extract_features(x)
          feats.append(feat)
          trues.append(y.cpu().numpy())
      X = np.vstack(feats)
      y = np.concatenate(trues)

      # PCA e SVM
      pca = PCA(n_components=pca_components)
      X_pca = pca.fit_transform(X)
      print(f"[INFO] Dimensão original: {X.shape[1]}")
      print(f"[INFO] Dimensão reduzida: {X_pca.shape[1]}")
      svm = SVC(kernel='rbf', C=svm_C, probability=True)
      svm.fit(X_pca, y)

      # Salva os modelos
      joblib.dump(pca, pca_path)
      joblib.dump(svm, svm_path)

      # Avaliação
      y_pred = svm.predict(X_pca)
      acc = accuracy_score(y, y_pred)
      prec = precision_score(y, y_pred, zero_division=0)
      rec = recall_score(y, y_pred, zero_division=0)
      f1 = f1_score(y, y_pred, zero_division=0)

      # Salvar CSV com métricas
      os.makedirs('output/resultados', exist_ok=True)
      svm_csv_path = os.path.join('output/resultados', 'svm_training_metrics.csv')
      with open(svm_csv_path, mode='w', newline='') as f:
          writer = csv.writer(f)
          writer.writerow(['accuracy', 'precision', 'recall', 'f1_score'])
          writer.writerow([acc, prec, rec, f1])

      return svm


    # def evaluate(self,
    #              loader: DataLoader,
    #              device: str = 'cpu',
    #              pca_path: str = 'output/Hybrid/pca.joblib',
    #              svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
    #     """
    #     Avalia o pipeline CNN→LSTM→PCA→SVM usando artefatos salvos.
    #     """
    #     pca = joblib.load(pca_path)
    #     svm = joblib.load(svm_path)
    #     feats, trues = [], []
    #     for x, y in loader:
    #         feat = self.extract_features(x)
    #         feats.append(feat)
    #         trues.append(y.cpu().numpy())
    #     X = np.vstack(feats)
    #     y_true = np.concatenate(trues)
    #     X_pca  = pca.transform(X)
    #     y_pred = svm.predict(X_pca)

    #     print(classification_report(y_true, y_pred))
    #     print("Accuracy:", accuracy_score(y_true, y_pred))
    
    def evaluate(self,
             loader: DataLoader,
             device: str = 'cpu',
             pca_path: str = 'output/Hybrid/pca.joblib',
             svm_path: str = 'output/Hybrid/hybrid_svm.joblib'):
      """
      Avalia o pipeline CNN→LSTM→PCA→SVM e gera métricas, gráficos e arquivos.
      """

      self.to(device).eval()
      pca = joblib.load(pca_path)
      svm = joblib.load(svm_path)
      feats, trues = [], []

      for x, y in loader:
          feat = self.extract_features(x)
          feats.append(feat)
          trues.extend(y.cpu().numpy())

      X = np.vstack(feats)
      y_true = np.array(trues)
      X_pca = pca.transform(X)
      y_pred = svm.predict(X_pca)
      y_proba = svm.predict_proba(X_pca)[:, 1]

      acc = accuracy_score(y_true, y_pred)
      prec = precision_score(y_true, y_pred)
      rec = recall_score(y_true, y_pred)
      f1 = f1_score(y_true, y_pred)
      fpr, tpr, _ = roc_curve(y_true, y_proba)
      roc_auc = auc(fpr, tpr)
      report = classification_report(y_true, y_pred, digits=4)
      precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)

      result_dir = os.path.join('output', 'resultados', 'Hybrid')
      os.makedirs(result_dir, exist_ok=True)

      # Terminal
      print(f"Acurácia:  {acc:.4f}")
      print(f"Precisão:  {prec:.4f}")
      print(f"Recall:    {rec:.4f}")
      print(f"F1-Score:  {f1:.4f}")
      print(f"AUC:       {roc_auc:.4f}")

      # Salvar relatório de classificação
      with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
          f.write("=== Classification Report ===\n")
          f.write(report)
          f.write("\n\n")
          f.write(f"Acurácia:       {acc:.4f}\n")
          f.write(f"Precisão:       {prec:.4f}\n")
          f.write(f"Recall:         {rec:.4f}\n")
          f.write(f"F1-Score:       {f1:.4f}\n")
          f.write(f"AUC:            {roc_auc:.4f}\n")

      # Salvar métricas em JSON
      with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
          json.dump({
              'acuracia': acc,
              'precisao': prec,
              'recall': rec,
              'f1_score': f1,
              'auc': roc_auc
          }, f, indent=4)

      # Matriz de confusão
      cm = confusion_matrix(y_true, y_pred)
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 13})
      plt.xlabel('Predito', fontsize=14)
      plt.ylabel('Real', fontsize=14)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'Hybrid_matriz_confusao.pdf'))
      plt.close()

      # Precision vs Recall
      plt.figure()
      plt.plot(recall_arr, precision_arr, marker='.')
      plt.xlabel('Recall', fontsize=14)
      plt.ylabel('Precision', fontsize=14)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'Hybrid_precision_vs_recall.pdf'))
      plt.close()

      # ROC Curve
      plt.figure()
      plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
      plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
      plt.xlabel('False Positive Rate', fontsize=14)
      plt.ylabel('True Positive Rate', fontsize=14)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.legend()
      plt.tight_layout()
      plt.savefig(os.path.join(result_dir, 'Hybrid_roc_curve.pdf'))
      plt.close()


