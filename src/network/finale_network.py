

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import copy
import numpy as np

# ======================= Définition du modèle =========================
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)

        self.dropout_rate = 0.25
        self.pool = nn.MaxPool2d(2, 2)

        # Calcul de la taille finale après convolutions et pooling pour une image 128x128
        self._to_linear = self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 100)
        self.fc2 = nn.Linear(100, 2)

    def _get_conv_output(self):
        x = torch.zeros(1, 3, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        return x.numel()

    def forward(self, X):
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# ======================= Transforms =========================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# ======================= Préparation des données =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = datasets.ImageFolder('/Users/OlivierDJEZVEDJIAN/Documents/Centrale/projetS6/ia-detection-leucemie/reseaux_manu/image_cancer/C-NMC_training_data/fold_0', transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ======================= Fonctions auxiliaires =========================
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in tqdm(dataset_dl, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        run_loss += loss_b * xb.size(0)
        t_metric += metric_b

    return run_loss / len_data, t_metric / len_data

def train_val(model, params, verbose=False):
    epochs = params["epochs"]
    opt = params["optimiser"]
    loss_func = params["f_loss"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in tqdm(range(epochs), leave=False):
        current_lr = get_lr(opt)
        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}, lr={current_lr:.6f}')

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("→ Best model saved")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                print("→ Learning rate adjusted. Reloading best weights.")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Accuracy: {100*val_metric:.2f}%")
            print("-" * 30)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


# ======================= Initialisation du modèle =========================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = Network().to(device)

    loss_func = nn.NLLLoss()
    opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
# ======================= Entraînement du modèle =========================
    params_train = {
    "train": train_dl,
    "val": val_dl,
    "epochs": 100,
    "optimiser": opt,
    "lr_change": lr_scheduler,
    "f_loss": loss_func,
    "weight_path": "weights_custom.pt",
}

    model, loss_hist, metric_hist = train_val(cnn_model, params_train, verbose=True)

# ======================= Évaluation sur le test set =========================
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n🎯 Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "HEMAI.pth")
    print("Modèle sauvegardé sous HEMAI.pth")

# ======================= Courbes et matrice de confusion =========================
    sns.set(style='whitegrid')
    epochs = params_train["epochs"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(x=list(range(1, epochs+1)), y=loss_hist["train"], ax=ax[0], label='Train Loss')
    sns.lineplot(x=list(range(1, epochs+1)), y=loss_hist["val"], ax=ax[0], label='Val Loss')
    ax[0].set_title("Loss")

    sns.lineplot(x=list(range(1, epochs+1)), y=metric_hist["train"], ax=ax[1], label='Train Accuracy')
    sns.lineplot(x=list(range(1, epochs+1)), y=metric_hist["val"], ax=ax[1], label='Val Accuracy')
    ax[1].set_title("Accuracy")

    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.show()

    