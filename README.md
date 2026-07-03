# Reseaux-de-neurones-a-3-couches
# Documentation du Projet : Réseau de Neurones Artificiels *From Scratch*

Ce document résume le travail accompli dans le fichier `fonctions.py` pour la création d'un Perceptron Multicouche (réseau de neurones artificiels) implémenté entièrement à partir de zéro (*From Scratch*) avec NumPy, appliqué à un problème de classification binaire.

---

## 1. Architecture du Réseau
Le modèle implémenté est un réseau de neurones à **3 couches** (2 couches cachées et 1 couche de sortie) :
- **Couche d'entrée ($C_0$)** : Reçoit les caractéristiques des données ($n_0$ variables).
- **Première couche cachée ($C_1$)** : $n_1$ neurones, activation Sigmoïde.
- **Deuxième couche cachée ($C_2$)** : $n_2$ neurones, activation Sigmoïde.
- **Couche de sortie ($C_3$)** : 1 neurone (classification binaire), activation Sigmoïde.

---

## 2. Synthèse des Fonctions Développées

### 📜 1. Initialisation des Paramètres (`init`)
Les poids ($W$) et les biais ($b$) de chaque couche sont initialisés de manière optimisée :
- **Biais ($b$)** : Initialisés à zéro (`np.zeros`) pour éviter de fausser les activations initiales.
- **Poids ($W$)** : Initialisés selon la méthode de **He (He-Normal / Kaiming)** en utilisant `np.random.randn(...) * np.sqrt(2 / n_in)`. Cette technique est cruciale pour maintenir la variance des activations et des gradients stable à travers les couches profondes, évitant le problème de disparition ou d'explosion du gradient.

### 🔄 2. Propagation Avant (`forward_prop`)
Cette fonction calcule les activations couche par couche en combinant la transformation linéaire et la fonction d'activation :
1. $\mathbf{Z_1} = \mathbf{W_1} \mathbf{X} + \mathbf{b_1} \longrightarrow \mathbf{A_1} = \sigma(\mathbf{Z_1})$
2. $\mathbf{Z_2} = \mathbf{W_2} \mathbf{A_1} + \mathbf{b_2} \longrightarrow \mathbf{A_2} = \sigma(\mathbf{Z_2})$
3. $\mathbf{Z_3} = \mathbf{W_3} \mathbf{A_2} + \mathbf{b_3} \longrightarrow \mathbf{A_3} = \sigma(\mathbf{Z_3})$

*Toutes les variables intermédiaires ($\mathbf{Z}$ et $\mathbf{A}$) sont mémorisées dans un dictionnaire pour être réutilisées lors de la rétropropagation.*

### 📉 3. Rétropropagation des Gradients (`back_prop`)
Elle applique l'algorithme de descente de gradient via la **règle de dérivation en chaîne (Chain Rule)** pour évaluer la contribution de chaque paramètre à l'erreur globale.
- Le calcul de l'erreur sur les couches cachées utilise la dérivée exacte de la fonction Sigmoïde exprimée sous forme stabilisée : $\sigma'(Z) = A 	imes (1 - A)$.
- Les gradients des biais ($\mathbf{db}$) et des poids ($\mathbf{dw}$) sont moyennés sur l'ensemble du batch (division par $m$).

### 🆙 4. Mise à jour des Paramètres (`MAJ`)
Met à jour les matrices de poids et de biais en soustrayant le gradient multiplié par le taux d'apprentissage ($lpha$ ou `learning_rate`) :
$$\mathbf{W} = \mathbf{W} - lpha \cdot \mathbf{dW}$$
$$\mathbf{b} = \mathbf{b} - lpha \cdot \mathbf{db}$$

### 🔮 5. Prédiction (`predict`)
Effectue une propagation avant complète sur des données de test ou de validation, extrait la probabilité finale $\mathbf{A_3}$ et applique un seuil strict à `0.5` pour retourner des classes binaires entières (`0` ou `1`).

### 🏋️‍♂️ 6. Boucle d'Entraînement (`fit`)
La fonction maîtresse du script orchestrant l'apprentissage :
- **Suivi des métriques** : Calcule et stocke à chaque itération la fonction de coût Log-Loss ainsi que la précision (`accuracy_score`) sur l'ensemble d'entraînement et de test.
- **Sécurité Numérique (Clipping)** : Utilisation de `np.clip(A, 1e-15, 1 - 1e-15)` pour sécuriser le calcul du `np.log` de la perte et bloquer les erreurs de type `divide by zero encountered in log`.
- **Visualisation** : Génère automatiquement en fin d'exécution deux graphiques côte à côte avec `matplotlib` : l'un pour la convergence des fonctions de coûts et l'autre pour l'évolution de la précision des ensembles Train et Test.

---

## 3. Améliorations et Corrections Majeures Apportées
Au fil du développement, plusieurs optimisations clés ont été implémentées pour garantir la stabilité et la justesse mathématique du modèle :
1. **Correction des dimensions des Biais** : Utilisation de `np.sum(..., axis=1, keepdims=True)` dans la rétropropagation pour maintenir les dimensions sous forme de vecteurs colonnes et éviter les comportements de broadcasting erronés de NumPy.
2. **Flattening pour l'Accuracy** : Aplatissement systématique des matrices (`.flatten()`) lors de l'envoi des prédictions à `accuracy_score` de Scikit-Learn pour assurer une comparaison ligne par ligne et non globale.
3. **Stabilité Mathématique** : Implémentation du clipping des probabilités pour éliminer les instabilités asymptotiques du logarithme.
