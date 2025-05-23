import os
import torch
from sklearn.metrics import accuracy_score

class TrainerCNN_2D:
    def __init__(self, dir_save: str, num_epochs: int, min_acc: float = 0.80, lr: float = 0.0011):
    
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

        for epoch in range(1, self.num_epochs + 1):
            # --- Treinamento ---
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            # --- Validação ---
            model.eval()
            val_loss = 0.0
            y_true, y_pred = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    _, preds = torch.max(outputs, dim=1)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.cpu().tolist())

            avg_val_loss = val_loss / len(test_loader.dataset)
            acc = accuracy_score(y_true, y_pred)

            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"Val Loss:   {avg_val_loss:.4f} "
                  f"Accuracy:   {acc:.4f}")

            # --- Salvar o melhor modelo ---
            if avg_val_loss < best_loss and acc >= self.min_acc:
                best_loss = avg_val_loss
                save_path = os.path.join(
                    self.dir_save,
                    f"CNN_Epoca-{epoch}_Acc-{acc:.2f}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss
                }, save_path)
                print(f"🔖 Melhor modelo salvo em: {save_path}")

        print("✅ Treinamento concluído.")
