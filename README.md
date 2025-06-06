## Name : HemAI

## Description
Créer une intelligence artificielle de reconnaissance d’image pour diagnostiquer la leucémie lymphoblastique à partir de frotti sanguin.


# IA detection leucémie

Contexte
La leucémie lymphoblastique à cellule B (B-alll) est un des cancers les plus répandus. Son diagnostique nécessite des tests invasifs et coûteux. 

L’examen microscopique du sang est une méthode courante de dépistage, mais il présente des limites lorsqu’il est réalisé manuellement par le personnel médical. (subjectivité, temps, accès limité...)

# Utilisation de l’IA
Pendant ce temps, les techniques d’intelligence artificielle (IA) permettent d’obtenir des résultats remarquables dans l’analyse d’images de microscopie sanguine.


## Structure du Projet

```
.
├── app/                      # Interface utilisateur
│   └── Interface_Cellular.py
├── data/                     # Données d'entrée (ex: images)
│   └── image_before_process.png
├── models/                   # Modèles exportés et fichiers de configuration
│   ├── cancer_cell_classification.pth
│   └── HEMAI_finale_model.pth
├── src/
│   ├── network/              # Définition des architectures de réseaux
│   │   ├── finale_network.py
│   │   ├── loader.py
│   │   └── modelV1.py
│   └── preprocessing/        # Scripts de prétraitement d'image
│       ├── binary_thresholding.py
│       ├── K_clustering.py
│       ├── masque.py
│       ├── resizing_224x224.py
│       ├── Retour_RGB.py
│       ├── RGB_LAB.py
│       └── Test_traitement_image.py
```

## Réseaux de Neurones

Les définitions des réseaux de neurones sont situées dans le dossier [`src/network`](./src/network). Vous y trouverez différentes versions d’architectures telles que :

- `finale_network.py` : architecture finale utilisée pour la prédiction
- `modelV1.py` : version expérimentale
- `loader.py` : fonctions pour charger les modèles et données

## Modèles Préentraînés

Les modèles préentraînés au format `.pth` sont disponibles dans le dossier [`models`](./models). Ils peuvent être directement chargés pour réaliser des prédictions sans avoir à entraîner les réseaux depuis zéro.

## Prétraitement

Le dossier [`src/preprocessing`](./src/preprocessing) contient tous les scripts de traitement d’images nécessaires avant l'inférence. Cela inclut :

- redimensionnement d’image (`resizing_224x224.py`)
- conversion de couleurs (`RGB_LAB.py`, `Retour_RGB.py`)
- seuillage, masquage et clustering (`binary_thresholding.py`, `masque.py`, `K_clustering.py`)


## Auteurs

- **Tom Connery**
- **Emmanuel Djevedjian**
- **Maté Cortela**
- **Antonin Decouvelaere**