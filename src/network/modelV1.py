import torch




##### attention à bien organiser les images comme ca ##################

# dataset/
# ├── healthy/ (HEM)
# │   ├── image1.jpg
# │   ├── image2.jpg
# │   └── ...
# └── cancerous/ (All)
#     ├── image1.jpg
#     ├── image2.jpg
#     └── ...

#########################################################################


from loader import data_loader

train_loader,test_loader=data_loader

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
optimizer = optim.Adam(model.parameters(), lr=0.001) # défintion de la fonction de perte et l'optimiseur 

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
