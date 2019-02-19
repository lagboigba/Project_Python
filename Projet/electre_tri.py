# Poids associés aux critères
from pprint import pprint


# Calcul cj

def concordance_partiel(type,h,bi,j):
    """

    :param h: Couche-culotte (donc liste)
    :param bi: profil (frontière qui YYYY)
    :param j: critère
    :return: 1 si H est au-moins aussi bon que bi, 0 sinon
    """
    #print('concord partielle')
    if(type == 'max'):
        #print('h[j] :',h[j])
        #print('bi[j] :',bi[j])

        if h[j] >= bi[j]:
            return 1
        else :
            return 0
    elif(type=='min'):
        if h[j] <= bi[j]:
            return 1
        else :
            return 0
    else:
        #print("Calcul concordance partielle impossible car type inconnu : ", type)
        return -1


def concordanceGlobal(h,bi,poids,type):
    """
    :param type: vecteur contenant ['min'/'max'] pour chaque critere
    :param h: Couche culotte (liste)
    :param bi: profil (frontière qui YYYY)
    :param poids: vecteur contenant poids pour chaque critere
    :return: indice de concordance global, -1 si erreur
    """
    #print('concordanceGlobal')
    numerateur = 0.0
    denom = 0.0
    # Pour chaque critère j
    #print('poids',poids)
    #print('len(h)-1 = ',len(h))
    for j in range(1,len(h)):
        #print("critère ",j)
        numerateur += poids[j-1]*concordance_partiel(type[j-1],h,bi,j) # len(poids) = len(h)-2
        #TODO : si c'est correct, passer tout en list comprehension pour lisibilite
        #print('numerateur',numerateur)
        denom += poids[j-1]
        #print('denom', denom)
        #print('----------------')
    return numerateur/denom


def Surclasse(seuilMajorite,h,bi,poids,type):
    """

    :param type: vecteur contenant ['min'/'max'] pour chaque critere
    :param seuilMajorite:
    :param h:
    :param bi:
    :param poids:
    :return:
    """

    if concordanceGlobal(h,bi,poids,type) > seuilMajorite:
        #print('Surclassé')
        return True
    else:
        #print('PAS Surclassé')
        return False


def AffectationOptimiste(h,classement,poids,type,seuil):
    """

    :param h:
    :param classement:
    :param poids:
    :param type:
    :param seuil:
    :return:
    """
    profil = len(classement)-1
    #print('PROFIL : ',profil)
    #print('H sur profil :', Surclasse(seuil, h, classement[profil], poids, type))
    #print('profil sur H : ', Surclasse(seuil, classement[profil], h, poids, type))
    while (Surclasse(seuil, h, classement[profil], poids, type)) and not(Surclasse(seuil, classement[profil], h, poids, type)):
        profil -= 1
    return profil


def AffectationPessimiste(h,classement,poids,type,seuil):
    """

    :param h:
    :param classement:
    :param poids:
    :param type:
    :param seuil:
    :return:
    """
    profil = 0
    while not (Surclasse(seuil, h, classement[profil], poids, type)):
        profil += 1
    return profil

def EvalOptimiste(lesCouches,classement,poids,type,seuil):
    """

    :param lesCouches:
    :param classement:
    :param poids:
    :param type:
    :param seuil:
    :return:
    """
    # pour chaque element de la matrice (une couche culotte), en partant de l'indice max (matrice profils[i][0]), on....
    for couche in range(0,len(lesCouches)): # len(lesCouches)+1 ?
        print("Couche "+str(lesCouches[couche][0])+" :",classement[AffectationOptimiste(lesCouches[couche],classement,poids,type,seuil)])
    return None

def EvalPessimiste(lesCouches,classement,poids,type,seuil):
    """

    :param lesCouches:
    :param classement:
    :param poids:
    :param type:
    :param seuil:
    :return:
    """
    liste = []
    for couche in range(0,len(lesCouches)): # len(lesCouches)+1 ?
        liste.append(["Couche "+str(couche),AffectationPessimiste(lesCouches[couche],classement,poids,type,seuil)])
        print("Couche " + str(lesCouches[couche][0]) + " :",classement[AffectationPessimiste(lesCouches[couche], classement, poids, type, seuil)])
    return liste


##
# main
##
import pandas as pd
import csv

poids = [3/5,2/5]
types = ['max','max']
SEUIL = 0.5
mat = []
matrice_profils = [
    ['Profil 6',4,4],
    ['Profil 5',3,3],
    ['Profil 4',2,2],
    ['Profil 3',1,1],
    ['Profil 2',-1,-1],
    ['Profil 1',-3,-3]
]

with open('mat.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    #header = next(reader)
    # on parse le fichier, en précisant les types pour chaque colonne
    #rows = [header] + [[row[0], int(row[1]), int(row[2]), float(row[3])] for row in reader]
    #mat = [[row[0], int(row[1]), int(row[2]), float(row[3])] for row in reader]
    mat = [[row[0], int(row[1]), int(row[2])] for row in reader]


# print("Données des couches mat : ",mat)
# print("Les profils :",matrice_profils)
# print("Les poids :",poids)
# print("Le type de notation",types)

print("#### DEBUT ELECTRE TRI ###")
print("optimiste")
EvalOptimiste(mat,matrice_profils,poids,types,SEUIL)

print("pessimiste")
#EvalOptimiste(mat[0:2],matrice_profils,poids,types,SEUIL)
EvalPessimiste(mat,matrice_profils,poids,types,SEUIL)
#
# df = pd.DataFrame(mat)
# df.columns=['Nom','Performance','Composition','NoteFinale']
# pprint(df)

# Surclasse(0.5, matrice_profils[5], mat[0:1], poids, types)