import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay


def init(n0,n1,n2):
    ## n0 represente les entrées du réseaux
    ## n1 le nombre de neurones dans la première couche
    ## n2 le nombre dans la deuxieme
    np.random.seed(42)
    w1 = np.random.randn(n1,n0)*np.sqrt(2/n0) ## la matrice des parametres pour les neurones de la premiere couche
    b1 = np.zeros((n1,1)) ## le vecteur des biais des neurones de la premiere couche

    w2= np.random.randn(n2,n1)*np.sqrt(2/n1) ## pareil pour la deuxieme couche et la troisieme(neurone de sortie)
    b2 = np.zeros((n2,1))

    w3 = np.random.randn(1,n2)*np.sqrt(2/n2)
    b3 = np.zeros((1,1))

    parametres = {"W1":w1,"W2":w2,"W3":w3,"b1":b1,"b2":b2,"b3":b3}
    return parametres


def forward_prop(parametres,X):
    ## La je calcule les scores et les activations pour chaque neuronne de chaque couche
    z1 = parametres["W1"]@X + parametres["b1"]
    A1 = np.maximum(0,z1) ## La on a la sortie de la premiere couche du réseaux
                            ## cette sortie represente les entrées de la seconde couche

    z2 = parametres["W2"]@A1 + parametres["b2"]
    A2 = np.maximum(0,z2)## la sortie de cette est également l'entrée du dernier neurone
    
    z3 = parametres['W3']@A2 + parametres["b3"]
    A3 = 1/(1+np.exp(-z3))

    activations = {"Z1":z1,"A1":A1,"Z2":z2,"A2":A2,"Z3":z3,"A3":A3}

    return activations



def back_prop(X,y,activations,parametres):
    ## Je calcule le gradient de la fonction cout par rapport aux parametres "w" et "b". Pour cela je passe par la derivation en chaine
    ## C'est la dérivation par chaine qui donne les facteurs "dz"
    dz3 = (1/X.shape[1])*(activations["A3"]-y)
    dw3 = dz3 @ activations["A2"].T
    db3 = np.sum(dz3,axis = 1,keepdims=True)

    dz2 = (parametres["W3"].T@dz3)*(activations["Z2"]>0)
    dw2 = dz2@activations["A1"].T
    db2 = np.sum(dz2,axis=1,keepdims=True)

    
    dz1 = (parametres["W2"].T@dz2)*(activations["Z1"]>0)
    dw1 = dz1@X.T
    db1 = np.sum(dz1,axis=1,keepdims=True)

    grads = {"dw3":dw3,"db3":db3,"dw2":dw2,"db2":db2,"dw1":dw1,"db1":db1}
    return grads


def MAJ(parametres,grads,learning_rate):
    ## Une fois que j'ai les gradients, il ne me reste qu'à mettre à jour mes parametres(me diriger dans la direction qui minimise la perte)

    parametres['W1'] = parametres['W1'] - learning_rate*grads["dw1"]
    parametres["b1"] = parametres["b1"] - learning_rate*grads['db1']

    parametres["W2"] = parametres["W2"] - learning_rate*grads["dw2"]
    parametres["b2"] = parametres["b2"] - learning_rate*grads['db2']

    parametres['W3'] = parametres["W3"] - learning_rate*grads["dw3"]
    parametres["b3"] = parametres["b3"] - learning_rate*grads['db3']

    return parametres


def predict(X,parametres):
    activations = forward_prop(parametres,X)
    A3 = activations["A3"]
    return (A3>=0.5)


def fit(X_train,X_test,y_train,y_test,n1,n2,learning_rate,n_iter):
    ## La j'entraine mon réseaux
    n0 = X_train.shape[0] ## n0 represente le nombre de features(entrées du réseau)
    parametres = init(n0,n1,n2)## Je cree la matrice des paramètres de chaque couche du réseau
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    ## En entrainant le réseux il faut que j'observe s'il apprend bien ou s'il fait de l'overfitting ou de l'underfitting
    for i in tqdm(range(n_iter)):
        activations_train = forward_prop(parametres,X_train)
        activations_test = forward_prop(parametres,X_test)
        

        if i %10 ==0:


            # On restreint les valeurs de A3 pour éviter les log(0)
            eps = 1e-15
            A3_train = np.clip(activations_train["A3"], eps, 1 - eps)## on plafonne nos valeurs avec clip, pour eviter d'avoir log(0)
            A3_test = np.clip(activations_test["A3"], eps, 1 - eps)
            perte_train = (-1/X_train.shape[1])*np.sum(y_train*np.log(A3_train)+ (1-y_train)*np.log(1-A3_train))
            perte_test = (-1/X_test.shape[1])*np.sum(y_test*np.log(A3_test) +(1-y_test)*np.log(1-A3_test))
            loss_train.append(perte_train)
            loss_test.append(perte_test)

            a1 = accuracy_score(y_train.flatten(),predict(X_train,parametres).flatten())
            a2 = accuracy_score(y_test.flatten(),predict(X_test,parametres).flatten())
            acc_train.append(a1)
            acc_test.append(a2)


        grads = back_prop(X_train,y_train,activations_train,parametres) ## calcul des gradients
        parametres = MAJ(parametres,grads,learning_rate) ## MAJ des gradients
    
    plt.figure(figsize=(17,10))
    ## j'affiche la convergence du réseaux sur les données d'entrainement et de test
    plt.subplot(1,2,1)
    plt.plot(loss_train,label="Ensemble_train",color="black")
    plt.plot(loss_test,label="Ensemble_test",color = "yellow")
    plt.xlabel("itérations")
    plt.ylabel("Perte/coût")
    plt.legend(loc='best')
    
    
    ## j'affiche l'accuracy du réseaux sur les données d'entrainement et de test
    plt.subplot(1,2,2)
    plt.plot(acc_train,label="Ensemble_train",color='black')
    plt.plot(acc_test,label="Ensemble_test",color='yellow')
    plt.xlabel("itérations")
    plt.ylabel("accuracy")
    plt.legend(loc='best')

    plt.show()
    
    return parametres