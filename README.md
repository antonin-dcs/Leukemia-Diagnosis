## Name
HemAI

## Description
Créer une intelligence artificielle de reconnaissance d’image pour diagnostiquer la leucémie lymphoblastique à partir de frotti sanguin.


# IA detection leucémie

Contexte
La leucémie lymphoblastique à cellule B (B-alll) est un des cancers les plus répandus. Son diagnostique nécessite des tests invasifs et coûteux. 

L’examen microscopique du sang est une méthode courante de dépistage, mais il présente des limites lorsqu’il est réalisé manuellement par le personnel médical. (subjectivité, temps, accès limité...)

Utilisation de l’ia
Pendant ce temps, les techniques d’intelligence artificielle (IA) permettent d’obtenir des résultats remarquables dans l’analyse d’images de microscopie sanguine.



# Etapes du traitement de l'image : 
1 : Conversion RGB-->LAB (Maté doit ajouter le fichier)
2 : formatage des images en tailles 224 x 224 resizing fonctionalité (déja merge dans le main par Antonin)
3 : K-clustering --> à creuser,
4: Binary Thresholding
5 : Création du masque
6 : Nettoyage : Comparaison avec l’image originale pour garder uniquement les éléments importants


Images Initiales 
Format : JPG RGB
Taille des fichiers : Ajustée pour être compatible avec le réseau CNN
Origine : Images provenant des hôpitaux de Téhéran


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.


## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.