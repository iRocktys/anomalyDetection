# models/Hybrid/TrainerHybridAttnSVM.py

import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

class TrainerHybridAttnSVM:
    def __init__(self,
                 dir_save: str,
                 num_epochs: int,
                 C: float = 10.0,
                 margin: float = 0.5,
                 lr: float = 1e-4):
        self.dir_save   = dir_save
        self.num_epochs = num_epochs
        self.C          = C
        self.margin     = margin
        self.lr         = lr
        os.makedirs(self.dir_save, exist_ok=True)

    def fit(self, model, train_loader, test_loader, device):
        model = model.to(device)
        criterion = nn.MultiMarginLoss(margin=self.margin)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=1.0/self.C
        )

        best_loss = float('inf')
        for epoch in range(1, self.num_epochs+1):
            model.train()
            tr_loss = 0.0
            for X,y in train_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss   = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()*X.size(0)
            avg_tr = tr_loss/len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            ys, ps = [], []
            with torch.no_grad():
                for X,y in test_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    val_loss += criterion(logits, y).item()*X.size(0)
                    ps.extend(logits.argmax(1).cpu().tolist())
                    ys.extend(y.cpu().tolist())
            avg_val = val_loss/len(test_loader.dataset)
            acc     = accuracy_score(ys, ps)

            print(f"[{epoch}/{self.num_epochs}] "
                  f"TrLoss={avg_tr:.4f} ValLoss={avg_val:.4f} ValAcc={acc:.4f}")

            if avg_val < best_loss:
                best_loss = avg_val
                ckpt = os.path.join(
                    self.dir_save,
                    f"HybridAttnSVM_Ep{epoch}_Val{avg_val:.4f}.pth"
                )
                torch.save(model.state_dict(), ckpt)
                print("🔖 Salvo:", ckpt)

        print("✅ Treino finalizado.")
