# Détection Géométrique de Main et Squelette (Sans Deep Learning)

Ce projet en Python illustre l'usage de la **vision par ordinateur classique** (sans réseau de neurones ni API pré-entraînée comme MediaPipe) pour isoler une main, calculer le centre géométrique de la paume et reconstruire un squelette virtuel à l'aide de la géométrie algorithmique et du traitement d'image.

## 🚀 Fonctionnalités

- **Traitement d'image pur** : Détection effectuée exclusivement via OpenCV et NumPy.
- **Normalisation de la luminosité** : Utilisation de l'égalisation d'histogramme adaptative (**CLAHE**) pour limiter l'impact des ombres.
- **Soustraction de fond adaptative** : Implémentation du modèle de mélange de gaussiennes (**MOG2**) pour isoler la main en mouvement.
- **Nettoyage morphologique** : Filtres d'ouverture et de fermeture pour éliminer le bruit binaire.
- **Extraction du squelette géométrique** :
  - **Centre de la paume** : Localisé grâce à la **Transformée de Distance** (`cv2.distanceTransform`).
  - **Sommets des doigts** : Extraits via l'**Enveloppe Convexe** (`cv2.convexHull`) et filtrage de proximité.
  - **Tracé dynamique** : Lignes reliant le centre de la paume à chaque extrémité.

---

## 🛠️ Prérequis

- Python 3.9 à 3.14
- Une webcam fonctionnelle

---

## 📦 Installation

1. **Cloner ou télécharger le dépôt** dans votre dossier de travail.

2. **Installer les dépendances nécessaires :**

   ```bash
   pip install opencv-python numpy