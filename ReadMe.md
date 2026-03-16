# LeNet-5

LeNet-5 est un réseau de neurones convolutionnel (CNN) pionnier développé par Yann LeCun, Léon Bottou, 
Yoshua Bengio et Patrick Haffner en 1998. Ce projet consiste à construire l'architecture de *LeNet-5 from scratch*.

---

## Architecture

```
Input (1, 32, 32)
    ↓
[C1]  Conv2D(1→6,   5x5) + Tanh   → (6, 28, 28)
[S2]  AvgPool(2x2)                 → (6, 14, 14)
[C3]  Conv2D(6→16,  5x5) + Tanh   → (16, 10, 10)
[S4]  AvgPool(2x2)                 → (16, 5, 5)
[C5]  Conv2D(16→120,5x5) + Tanh   → (120, 1, 1)
[F6]  Linear(120→84)    + Tanh    → (84,)
[Out] Linear(84→10)  + Softmax    → (10,)
```

---
## Structure du projet

```
LeNet-5_from_scratch/
├── Code_LeNet-5.ipynb     # Notebook principal contenant tout le code
├── README.md              # Ce fichier
└── Rapport_LeNet.pdf         # Rapport d'une page du projet
```

---

## Lancer le projet

### 1. Cloner le repo
```bash
git clone https://github.com/BGuiguiri/LeNet-5_from_scratch
cd LeNet-5_from_scratch
```

### 2. Installer les dépendances
```bash
pip install numpy matplotlib scikit-learn
```

### 3. Lancer Jupyter
```bash
jupyter notebook Code_LeNet-5.ipynb
```

### 4. Exécuter toutes les cellules dans l'ordre
Le notebook est organisé en étapes numérotées — exécuter de haut en bas.

---

## Résultats

| Modèle            | Test Accuracy | Epochs |
|-------------------|--------------|--------|
| LeNet-5 sans BN   | ~78%         | 10     |
| LeNet-5 avec BN   | ~83%         | 10     |
| **Gain**          | **+5%**      | —      |

---

## Composants implémentés from scratch

| Composant       | Fichier              | Description                          |
|-----------------|----------------------|--------------------------------------|
| `Conv2D`        | `Code_LeNet-5.ipynb` | Forward + Backward manuel            |
| `AvgPool2D`     | `Code_LeNet-5.ipynb` | Average pooling + gradient           |
| `Linear`        | `Code_LeNet-5.ipynb` | Fully connected + backprop           |
| `Tanh / ReLU`   | `Code_LeNet-5.ipynb` | Activations + dérivées               |
| `BatchNorm`     | `Code_LeNet-5.ipynb` | BN train/test + moyennes mobiles     |
| `Flatten`       | `Code_LeNet-5.ipynb` | Reshape + backward                   |
| `SGD`           | `Code_LeNet-5.ipynb` | Optimizer mise à jour des poids      |
| `CrossEntropy`  | `Code_LeNet-5.ipynb` | Loss + gradient Softmax combiné      |

---

## Choix techniques

### Initialisation : Xavier
```python
std = sqrt(2 / (fan_in + fan_out))
```
Adapté à Tanh : maintient la variance des activations stable entre les couches,
évitant la disparition ou l'explosion du gradient.

### Activation : Tanh
- Sorties centrées en 0 → gradients moins oscillants
- Dérivée max = 1 (vs 0.25 pour Sigmoid) → convergence 2-4x plus rapide
- Compatible Xavier pour garder z dans la zone active [-2, 2]

---

## Ablation Study

Comparaison LeNet-5 **avec** vs **sans** Batch Normalization sur MNIST :

- **BN accélère la convergence** : plateau atteint epoch 4-5 vs 7-8
- **BN améliore l'accuracy** : +5% sur le test set
- **BN stabilise** : courbes de loss plus lisses, moins d'oscillations

---

## Dépendances

```
numpy
matplotlib
scikit-learn   # uniquement pour fetch_openml (téléchargement MNIST)
```
---

*Enseignant : Rodéo Oswald Y. TOHA (Engineer in CV & GenAI)*
