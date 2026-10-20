# Détection de Squelette de Main, Comptage de Doigts & Effet de Proximité

Ce projet en Python utilise **OpenCV** et **MediaPipe Tasks (Hand Landmarker)** pour capturer le flux vidéo d'une webcam en temps réel, analyser la structure anatomique des mains, compter le nombre de doigts levés et générer un effet visuel dynamique de rapprochement entre les extrémités.

---

## 🚀 Fonctionnalités

- **Tracking en temps réel** : Capture fluide depuis la webcam avec retour miroir pour une ergonomie naturelle.
- **Détection multi-mains** : Prise en charge simultanée de 2 mains via l'API moderne `mediapipe.tasks`.
- **Squelette anatomique complet** : Tracé personnalisé des 21 repères (`landmarks`) et des 20 connexions articulaires.
- **Comptage dynamique des doigts** :
  - **Pouce** : Évaluation selon l'axe horizontal ($X$) adaptée à la latéralité de la main (`Left` / `Right`) compensée par l'effet miroir.
  - **Autres doigts** : Évaluation selon l'axe vertical ($Y$) en comparant la position de l'extrémité (`tip`) par rapport à l'articulation inférieure.
- **Reconnaissance de gestes** : Affiche **OK**, **PEACE**, **ROCK** (index et auriculaire levés), **MAIN OUVERTE**, **POUCE LEVÉ**, **POUCE BAISSÉ** et **POING** selon la position des doigts.
- **Effet visuel de proximité (Interaction)** :
  - Un trait dynamique relie les extrémités des doigts lorsqu'ils se rapprochent.
  - Un point d'impact central s'affiche et grossit à mesure que la distance diminue.
  - Transition dynamique de couleur (du **vert** au **rouge**) basée sur la distance euclidienne en pixels.
- **Indicateurs à l'écran** : Affichage en direct du nombre d'images par seconde (**FPS**) et du total des doigts levés.

---

## 🛠️ Prérequis

- **Python** : 3.9 à 3.12 (ou 3.14+ avec l'API Tasks)
- **Webcam** fonctionnelle

---

## 📦 Installation

1. **Cloner ou télécharger le dépôt** dans votre dossier de travail.

2. **Installer les dépendances nécessaires :**

   ```bash
   pip install opencv-python mediapipe numpy