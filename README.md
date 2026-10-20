# Détection de Squelette de Main et Comptage de Doigts

Ce projet en Python utilise **OpenCV** et **MediaPipe Tasks (Hand Landmarker)** pour capturer le flux vidéo d'une webcam en temps réel, détecter les repères anatomiques des mains et compter le nombre de doigts levés.

## 🚀 Fonctionnalités

- **Capture en temps réel** via la webcam avec effet miroir.
- **Détection multi-mains** (jusqu'à 2 mains simultanément) via la nouvelle API `mediapipe.tasks`.
- **Rendu graphique** des repères de la main.
- **Comptage dynamique des doigts** :
  - Traitement adapté du pouce en fonction de la latéralité de la main (`Left` / `Right`) sur l'axe X.
  - Traitement des 4 autres doigts sur l'axe Y (hauteur relative des extrémités).
- **Affichage dynamique** du total des doigts levés à l'écran.

---

## 🛠️ Prérequis

- Python 3.9 à 3.12 (ou version compatible avec MediaPipe)
- Une webcam fonctionnelle

---

## 📦 Installation

1. **Cloner ou télécharger le dépôt** dans votre dossier de travail.

2. **Installer les dépendances nécessaires :**

   ```bash
   pip install opencv-python mediapipe

---

## Lancer le projet

**Exécuter la commande**

```bash
python detection_squelette.py