# Détection de Squelette de Main, Comptage de Doigts & Effet de Proximité

Ce projet en Python utilise **OpenCV** et l'API moderne **MediaPipe Tasks (Hand Landmarker)** pour capturer le flux vidéo d'une webcam en temps réel, analyser la structure anatomique des mains, compter le nombre de doigts levés, reconnaître des gestes usuels et générer un effet visuel dynamique de rapprochement entre les extrémités des doigts.

---

## 🚀 Fonctionnalités

- **Tracking en temps réel** : Capture vidéo fluide depuis la webcam avec inversion miroir (`cv2.flip`) pour une interaction intuitive.
- **Détection multi-mains** : Prise en charge de 1 à 2 mains simultanément via l'API `mediapipe.tasks.python.vision.HandLandmarker`.
- **Squelette anatomique complet** : Tracé personnalisé des 21 repères (*landmarks*) et des 20 connexions articulaires de la main.
- **Comptage dynamique des doigts** :
  - **Pouce** : Évaluation sur l'axe horizontal ($X$) adaptée à la latéralité de la main (`Left` / `Right`) compensée par l'effet miroir.
  - **Autres doigts** : Évaluation sur l'axe vertical ($Y$) en comparant la position de l'extrémité (`tip`) par rapport à l'articulation inférieure.
- **Reconnaissance automatique des gestes** :
  - **OK** : Pincement du pouce et de l'index ($\le 35\%$ de la largeur de la paume) avec les autres doigts levés.
  - **PEACE** : Index et majeur levés.
  - **ROCK** : Index et auriculaire levés.
  - **MAIN OUVERTE** : Tous les doigts levés.
  - **POING** : Tous les doigts repliés vers la paume.
  - **POUCE LEVÉ** / **POUCE BAISSÉ** : Doigts repliés et pouce dirigé vers le haut ou le bas par rapport au poignet.
- **Effet visuel de proximité (Interaction)** :
  - Détection automatique de la distance euclidienne en pixels entre toutes les paires de bouts de doigts d'une main.
  - Tracé d'une ligne dynamique lorsque la distance est inférieure au seuil (65 px).
  - Épaisseur du trait et taille du point d'impact ajustées dynamiquement selon la proximité.
  - Transition de couleur BGR progressive du **vert** (éloigné) au **rouge** (très proche).
- **Affichage dynamique (HUD)** :
  - Fréquence d'images en temps réel (**FPS**).
  - Total cumulé des doigts levés à l'écran.
  - Étiquette du geste reconnu affichée au-dessus de chaque main détectée.

---

## 🛠️ Prérequis

- **Python** : 3.9 à 3.12 (ou version compatible avec MediaPipe Tasks)
- **Webcam** fonctionnelle

---

## 📦 Installation & Configuration

1. **Cloner ou télécharger le dépôt** dans votre dossier de travail.

2. **Installer les dépendances nécessaires :**

   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Télécharger le modèle MediaPipe Hand Landmarker :**
   Le script nécessite le fichier modèle officiel MediaPipe `hand_landmarker.task` situé dans le même répertoire que le script Python.

   - Téléchargez le modèle depuis le dépôt MediaPipe ou via ce lien direct : [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
   - Placez le fichier `hand_landmarker.task` à la racine du projet.

---

## 💻 Utilisation

Exécutez le script principal avec la commande suivante :

```bash
python detection_squelette_main.py
```

- **Contrôles** : Appuyez sur la touche `q` pour quitter la fenêtre d'affichage.

---

## 📁 Structure du Projet

```text
.
├── detection_squelette_main.py  # Script principal Python
├── hand_landmarker.task        # Modèle IA MediaPipe (à télécharger)
└── README.md                    # Documentation du projet
```

---

## 🧮 Logique de Détection et Calculs

- **Standardisation de la taille de la main** : Les seuils de gestes (comme le signe *OK* ou le *Poing*) sont calculés en fonction de la largeur relative de la paume (distance euclidienne entre le MCP de l'index et le MCP de l'auriculaire), garantissant une détection stable quelle que soit la distance de la main par rapport à la caméra.
- **Couleur dynamique de proximité** : La couleur du lien d'interaction est calculée via un ratio de distance $\text{ratio} = \min(1.0, \max(0.0, \frac{\text{dist}}{\text{THRESHOLD\_DIST}}))$, interpolant le composant Vert ($255 \times \text{ratio}$) et le composant Rouge ($255 \times (1.0 - \text{ratio})$).