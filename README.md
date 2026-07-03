# Documentation du Projet : Réseau de Neurones Artificiels *From Scratch*

Ce document résume le travail accompli dans le fichier `fonctions.py` pour la création d'un Perceptron Multicouche (réseau de neurones à plusieurs couches) implémenté entièrement à partir de zéro (*From Scratch*) avec NumPy, appliqué à un problème de classification binaire.

---

## 1. Architecture du Réseau
Le modèle implémenté est un réseau de neurones profond à **3 couches** (2 couches cachées et 1 couche de sortie) :
- **Couche d'entrée ($C_0$)** : Reçoit les caractéristiques des données ($n_0$ variables d'entrée).
- **Première couche cachée ($C_1$)** : $n_1$ neurones, fonction d'activation **ReLU**.
- **Deuxième couche cachée ($C_2$)** : $n_2$ neurones, fonction d'activation **ReLU**.
- **Couche de sortie ($C_3$)** : 1 neurone pour la classification binaire, fonction d'activation **Sigmoïde**.

---

## 2. Synthèse des Fonctions Développées

### 📜 1. Initialisation des Paramètres (`init`)
Les poids ($W$) et les biais ($b$) de chaque couche sont initialisés de manière optimisée :
- **Biais ($b$)** : Initialisés à zéro (`np.zeros`) sous forme de vecteurs colonnes pour éviter de biaiser les activations initiales.
- **Poids ($W$)** : Initialisés selon la méthode de **He (He-Normal / Kaiming)** en utilisant `np.random.randn(...) * np.sqrt(2 / n_in)`. Cette technique est particulièrement adaptée et indispensable lorsque l'on utilise la fonction d'activation **ReLU** afin d'éviter le problème de disparition ou d'explosion du gradient lors de l'apprentissage.

### 🔄 2. Propagation Avant (`forward_prop`)
Cette fonction calcule les activations couche par couche en combinant la transformation linéaire et les fonctions d'activation spécifiques :
1. $\mathbf{Z_1} = \mathbf{W_1} \mathbf{X} + \mathbf{b_1} \longrightarrow \mathbf{A_1} = \max(0, \mathbf{Z_1})$ *(ReLU)*
2. $\mathbf{Z_2} = \mathbf{W_2} \mathbf{A_1} + \mathbf{b_2} \longrightarrow \mathbf{A_2} = \max(0, \mathbf{Z_2})$ *(ReLU)*
3. $\mathbf{Z_3} = \mathbf{W_3} \mathbf{A_2} + \mathbf{b_3} \longrightarrow \mathbf{A_3} = \sigma(\mathbf{Z_3})$ *(Sigmoïde)*

*Toutes les variables intermédiaires ($\mathbf{Z}$ et $\mathbf{A}$) sont sauvegardées dans un dictionnaire de cache pour être exploitées lors de la rétropropagation.*

### 📉 3. Rétropropagation des Gradients (`back_prop`)
Elle applique l'algorithme de descente de gradient via la **règle de dérivation en chaîne (Chain Rule)**. Elle a été adaptée spécifiquement pour gérer la transition entre la Sigmoïde de sortie et la ReLU des couches cachées :
- Pour la couche de sortie ($C_3$), le calcul de l'erreur combiné avec la Log-Loss donne de manière exacte : $\mathbf{dZ_3} = \mathbf{A_3} - \mathbf{y}$.
- Pour les couches cachées ($C_2$ et $C_1$), le gradient est propagé en arrière en appliquant la dérivée de la fonction **ReLU** (qui vaut `1` si $Z > 0$ et `0` sinon) :
  - $\mathbf{dZ_2} = (\mathbf{W_3}^T \mathbf{dZ_3}) \odot \mathbb{I}(\mathbf{Z_2} > 0)$
  - $\mathbf{dZ_1} = (\mathbf{W_2}^T \mathbf{dZ_2}) \odot \mathbb{I}(\mathbf{Z_1} > 0)$
- Les gradients des biais ($\mathbf{db}$) utilisent `np.sum(..., axis=1, keepdims=True)` pour conserver leur structure vectorielle stricte.

### 🆙 4. Mise à jour des Paramètres (`MAJ`)
Modifie les matrices de poids et de biais en soustrayant le produit du gradient et du taux d'apprentissage ($ lpha$ ou `learning_rate`) :
$$\mathbf{W} = \mathbf{W} -  lpha \cdot \mathbf{dW}$$
$$\mathbf{b} = \mathbf{b} -  lpha \cdot \mathbf{db}$$

### 🔮 5. Prédiction (`predict`)
Effectue une propagation avant sur de nouvelles données, extrait la probabilité finale $\mathbf{A_3}$ et applique un seuil de décision strict à `0.5` pour retourner des classes binaires entières (`0` ou `1`).

### 🏋️‍♂️ 6. Boucle d'Entraînement (`fit`)
Fonction centrale qui orchestre l'apprentissage :
- **Sécurité Numérique (Clipping)** : Utilisation de `np.clip(A3, 1e-15, 1 - 1e-15)` pour encapsuler les probabilités et bloquer définitivement l'erreur asymptotique `divide by zero encountered in log` dans le calcul de la perte (Log-Loss).
- **Aplatissement (Flattening)** : Utilisation de `.flatten()` lors du calcul de la précision (`accuracy_score`) pour forcer une comparaison élément par élément plutôt qu'une comparaison matricielle globale erronée.
- **Visualisation** : Génère automatiquement deux graphiques de convergence (un pour la perte et un pour l'accuracy) comparant les performances d'entraînement et de test au fil des itérations.
