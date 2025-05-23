import os
import torch
from sklearn.metrics import accuracy_score

class TrainerHybrid:
    def __init__(self,
                 dir_save: str,
                 num_epochs: int,
                 min_acc: float = 0.80,
                 lr: float = 0.0011):
        """
        dir_save   -> pasta para salvar checkpoints
        num_epochs -> total de épocas
        min_acc    -> só salva modelo se acc_val ≥ min_acc
        lr         -> learning rate
        """
        self.dir_save   = dir_save
        self.num_epochs = num_epochs
        self.min_acc    = min_acc
        self.lr         = lr
        os.makedirs(self.dir_save, exist_ok=True)

    def fit(self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            test_loader:  torch.utils.data.DataLoader,
            device:       torch.device):
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_loss = float('inf')
        for epoch in range(1, self.num_epochs+1):
            # — Treino —
            model.train()
            loss_train = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * inputs.size(0)
            avg_train = loss_train / len(train_loader.dataset)

            # — Validação —
            model.eval()
            loss_val = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss_val += criterion(outputs, labels).item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())
            avg_val = loss_val / len(test_loader.dataset)
            acc = accuracy_score(y_true, y_pred)

            print(f"Epoch {epoch}/{self.num_epochs} "
                  f"Train Loss: {avg_train:.4f} "
                  f"Val Loss:   {avg_val:.4f} "
                  f"Val Acc:    {acc:.4f}")

            # — checkpoint —
            if avg_val < best_loss and acc >= self.min_acc:
                best_loss = avg_val
                path = os.path.join(
                    self.dir_save,
                    f"Hybrid_Ep{epoch}_Acc{acc:.2f}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }, path)
                print("🔖 Salvo:", path)

        print("✅ Treino concluído.")
