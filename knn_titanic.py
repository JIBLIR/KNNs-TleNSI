
# Importation des bibliothèques
import pandas as pd
from math import * 
import matplotlib.pyplot as plt
import math
import statistics
from statistics import mode
import numpy as np

# Importation des données
data = pd.read_csv(r"NNs from sractch\Class_Model\01_00_titanic.csv")

# Labelisation et suppression des données triviales comme le Ticket, qui n'apporte rien à l'analyse mais aussi le  nom, la cabine, l'id du passager et le lieu d'embarquement 
labels = data["Survived"]
data = data.drop(["Survived", "Ticket", "Cabin", "Name","PassengerId","Embarked"], axis=1) # .drop permet de suppremer les colonnes

#Encodage pour le sexe
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

data = data.fillna(data.mean()).astype(int) #Moyenne des données manquantes 

#Mélange des données 
data_melange = data.sample(frac=1, random_state=42).reset_index(drop=True) 
labels_melange = labels.loc[data_melange.index].reset_index(drop=True)

#Répartision de 20% et 80% 
test_taille = int(0.2 * len(data_melange))


test_x = data_melange[:test_taille]  #178 lignes pour faire des test 
test_y = labels_melange[:test_taille] 

train_x = data_melange[test_taille:] # 713 lignes or test 
train_y = labels_melange[test_taille:]

# on transforme les données en liste
data_list = train_x.values.tolist()
data_target = train_y.values.tolist()

def distance_euclidienne(x1):
    """ fonction distance euclidienne, adaptée pour toutes les dimensions"""
    # n'ayant pas trouvé d'autres moyens, nous avons défini cible en tant que variable globale car la fonction sorted est complexe à utiliser
    global cible
    somme = 0
    for i in range(len(x1)):
        somme += (x1[i] - cible[i])**2
    distance = math.sqrt(somme)
    return distance

def knn(X,k):
    """la fonction k plus proches voisins prend en entrée :
    - X : les données
    - k : le nombre de voisins à prendre en compte"""
    # on utilise là aussi une variable globale, qui est justifié car nous avons besoin de notre dataset 
    global data_target
    # on calcule la norme, donc la distane entre le point cible et les autre points
    X_sorted = sorted(X,key=distance_euclidienne)
    # on range notre liste, cela nous renvoit ce type de résultat [(0, 102), (3, 657), (4, 2643), (1, 4325), (2, 5125)]
    lst = list(enumerate(X_sorted))
    # donc on ne sélectionne que le deuxième élément qui nous intéresse c'est à dire x[1]
    lst.sort(key = lambda x: x[1])
    nearest_neighbors = []
    for i in range(k):
        nearest_neighbors.append(lst[i])
    nns_index = []
    # on sélectionne la ligne correspondant au voisin : nearest_neighbors[i][0] puis on map la classe correspondante : data_target[nearest_neighbors[i][0]]
    for i in range(len(nearest_neighbors)):
        nns_index.append(data_target[nearest_neighbors[i][0]])
    return nns_index

def mse(y_true,y_pred):
    """ on calcule l'erreur moyenne"""
    return (1/len(y_pred)) *  np.sum((y_true-y_pred)**2)

def comptage(nns_index):
    """permet de compter quelle est la classe la plus importante
    en calculant le mode : en statistique la valeur la plus fréquente"""
    return mode(nns_index)


test_x = test_x.values.tolist()
y_true = np.array(test_y.values.tolist())
y_pred = []
loss_list = []
# on parcoure toutes les valeurs de k jusqu'au nombre d'échantillon du test set
for k in range(1,len(test_x)):
    y_pred = []
    for i in range(len(test_x)):
        # on défini x_cible
        cible = test_x[i]
        # on calcule notre liste L de taille k
        nns_index = knn(data_list,k)
        # on fait apparaitre chaque valeur de y hat
        y_pred.append(comptage(nns_index))
    y_pred = np.array(y_pred)
    # on calcule le coût
    loss = mse(y_true,y_pred)
    loss_list.append(loss)

# on affiche le graphique 
iterations = np.arange(len(loss_list))
plt.plot(iterations,loss_list)
plt.show()
