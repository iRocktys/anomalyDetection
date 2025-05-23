import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

class TrainerHybridSVM:
    def __init__(self,
                 dir_save: str,
                 num_epochs: int,
                 C: float = 1.0,
                 margin: float = 1.0,
                 lr: float = 0.0011):
        """
        dir_save   -> pasta para salvar checkpoints
        num_epochs -> total de épocas
        C          -> trade‐off SVM (1/weight_decay)
        margin     -> margem do hinge‐loss
        lr         -> learning rate
        """
        self.dir_save   = dir_save
        self.num_epochs = num_epochs
        self.C          = C
        self.margin     = margin
        self.lr         = lr
        os.makedirs(self.dir_save, exist_ok=True)

    def fit(self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            test_loader:  torch.utils.data.DataLoader,
            device:       torch.device):
        model = model.to(device)
        criterion = nn.MultiMarginLoss(margin=self.margin)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1.0/self.C
        )

        best_val_loss = float('inf')
        for epoch in range(1, self.num_epochs+1):
            # treino
            model.train()
            total_tr = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                scores = model(X)
                loss   = criterion(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_tr += loss.item()*X.size(0)
            avg_tr = total_tr / len(train_loader.dataset)

            # validação
            model.eval()
            total_val = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    scores = model(X)
                    total_val += criterion(scores, y).item()*X.size(0)
                    preds = scores.argmax(dim=1)
                    y_true.extend(y.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())
            avg_val = total_val / len(test_loader.dataset)
            acc = accuracy_score(y_true, y_pred)

            print(f"[{epoch}/{self.num_epochs}] "
                  f"TrainLoss={avg_tr:.4f} "
                  f"ValLoss={avg_val:.4f} "
                  f"ValAcc={acc:.4f}")

            # salvar melhor
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                fname = f"HybridSVM_Ep{epoch}_Val{avg_val:.4f}.pth"
                path  = os.path.join(self.dir_save, fname)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, path)
                print("🔖 Salvo:", path)

        print("✅ Treino SVM‐loss concluído.")
