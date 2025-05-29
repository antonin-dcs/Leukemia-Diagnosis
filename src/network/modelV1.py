import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


##### attention à bien organiser les images comme ca ##################

# dataset/
# ├── healthy/
# │   ├── image1.jpg
# │   ├── image2.jpg
# │   └── ...
# └── cancerous/
#     ├── image1.jpg
#     ├── image2.jpg
#     └── ...

#########################################################################



# Transformations pour les images
transform = transforms.Compose([ 
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Charger les datasets
train_dataset = datasets.ImageFolder('D:/image_cancer/dataset', transform=transform) #transforme toute les images de la manière ci-dessus
test_dataset = datasets.ImageFolder(r'D:\image_cancer\fold_0', transform=transform) 

# Créer les DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # charge les images par lots de 32 en mélangeant les données
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # à chaque époque de l'entrainement

print("la taille du train_dataset est:", len(train_dataset))
print("la taille du test_dataset est:", len(test_dataset))

# Charger un lot d'images du train_loader
images, labels = next(iter(train_loader))

# Afficher la taille de la première image du lot
print("Taille de l'image :", images[0].shape)



import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):                          # La classe de base pour tous les modules de réseau de neurones. 
        super(SimpleCNN, self).__init__()        # En héritant de cette classe, nous pouvons définir notre propre architecture de réseau.
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):           # Cette méthode définit le passage en avant (forward pass) du réseau. 
        x = F.relu(self.conv1(x))   # Elle prend une entrée x et applique les couches définies dans l'ordre.
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()  # créé une distance de modèle défini
print(model)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001) # défintion de la fonction de perte et l'optimisateur 

num_epochs = 10

for epoch in range(num_epochs):              # entraine le modèle
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


model.eval()                                           
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

torch.save(model.state_dict(), 'cancer_cell_classifier_petit.pth') # permet de sauvegarder le modèle
