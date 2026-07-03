# Documentation du Projet : Réseau de Neurones Artificiels From Scratch

Ce document résume le travail accompli dans le fichier fonctions.py pour la création d'un Perceptron Multicouche (réseau de neurones à plusieurs couches) implémenté entièrement à partir de zéro (From Scratch) avec NumPy, appliqué à un problème de classification binaire.

---

## 1. Architecture du Réseau
Le modèle implémenté est un réseau de neurones profond à 3 couches (2 couches cachées et 1 couche de sortie) :
- Couche d'entrée : Reçoit les caractéristiques des données (variables d'entrée).
- Première couche cachée : Neurones activés par la fonction ReLU.
- Deuxième couche cachée : Neurones activés par la fonction ReLU.
- Couche de sortie : 1 neurone pour la classification binaire, activé par la fonction Sigmoïde.

---

## 2. Synthèse des Fonctions Développées

### 1. Initialisation des Paramètres (init)
Les poids (W) et les biais (b) de chaque couche sont initialisés de manière optimisée :
- Biais (b) : Initialisés à zéro sous forme de vecteurs colonnes pour éviter de biaiser les activations initiales.
- Poids (W) : Initialisés selon la méthode de He (He-Normal / Kaiming) en utilisant np.random.randn. Cette technique est indispensable lorsque l'on utilise la fonction d'activation ReLU afin d'éviter le problème de disparition ou d'explosion du gradient lors de l'apprentissage.

### 2. Propagation Avant (forward_prop)
Cette fonction calcule les activations couche par couche en combinant la transformation linéaire et les fonctions d'activation spécifiques :
1. Couche 1 : Z1 = W1 * X + b1  -->  A1 = ReLU(Z1)
2. Couche 2 : Z2 = W2 * A1 + b2  -->  A2 = ReLU(Z2)
3. Couche 3 : Z3 = W3 * A2 + b3  -->  A3 = Sigmoïde(Z3)

Toutes les variables intermédiaires (Z et A) sont sauvegardées dans un dictionnaire de cache pour être exploitées lors de la rétropropagation.

### 3. Rétropropagation des Gradients (back_prop)
Elle applique l'algorithme de descente de gradient via la règle de dérivation en chaîne (Chain Rule). Elle a été adaptée spécifiquement pour gérer la transition entre la Sigmoïde de sortie et la ReLU des couches cachées :
- Pour la couche de sortie, le calcul de l'erreur combiné avec la Log-Loss donne de manière exacte : dZ3 = A3 - y.
- Pour les couches cachées, le gradient est propagé en arrière en appliquant la dérivée de la fonction ReLU (qui vaut 1 si Z > 0 et 0 sinon).
- Les gradients des biais (db) utilisent np.sum(..., axis=1, keepdims=True) pour conserver leur structure vectorielle stricte.

### 4. Mise à jour des Paramètres (MAJ)
Modifie les matrices de poids et de biais en soustrayant le produit du gradient et du taux d'apprentissage (learning_rate) :
- W = W - learning_rate * dW
- b = b - learning_rate * db

### 5. Prédiction (predict)
Effectue une propagation avant sur de nouvelles données, extrait la probabilité finale A3 et applique un seuil de décision strict à 0.5 pour retourner des classes binaires entières (0 ou 1).

### 6. Boucle d'Entraînement (fit)
Fonction centrale qui orchestre l'apprentissage :
- Sécurité Numérique (Clipping) : Utilisation de np.clip(A3, 1e-15, 1 - 1e-15) pour encapsuler les probabilités et bloquer définitivement l'erreur divide by zero encountered in log dans le calcul de la perte (Log-Loss).
- Aplatissement (Flattening) : Utilisation de .flatten() lors du calcul de la précision (accuracy_score) pour forcer une comparaison élément par élément plutôt qu'une comparaison matricielle globale erronée.
- Visualisation : Génère automatiquement deux graphiques de convergence (un pour la perte et un pour l'accuracy) comparant les performances d'entraînement et de test au fil des itérations.
