from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Transformations pour les images
transform = transforms.Compose([ 
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

def data_loader():
# Charger les datasets
    train_dataset = datasets.ImageFolder('D:/image_cancer/dataset', transform=transform) #transforme toute les images de la manière ci-dessus
    test_dataset = datasets.ImageFolder('D:/image_cancer/fold_0', transform=transform) 

    # Créer les DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # charge les images par lots de 32 en mélangeant les données
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # à chaque epoch de l'entrainement
    return train_loader,test_loader


if __name__=='__main__':
    train_loader,_=data_loader()

    # Charger un lot d'images du train_loader
    images, labels = next(iter(train_loader))

    # Afficher la taille de la première image du lot
    print("Taille de l'image :", images[0].shape)
