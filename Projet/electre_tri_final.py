# Poids associés aux critères
from pprint import pprint

def concordance_partiel(type,h,bi,j):
    """

    :param h: 1 Couche-culotte (donc liste)
    :param bi: profil (frontière qui YYYY)
    :param j: critère
    :return: 1 si H est au-moins aussi bon que bi sur le critère j, 0 sinon
    """
    print('concord partielle')
    print("h[j] : ",h[j])
    print("bi[j] : ",bi[j])

    if(type == 'max'):
        if h[j] >= bi[j]:
            print("h[j] >= bi[j]")
            return 1
        else :
            print("h[j] < bi[j]")
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
    print(" ~~~~~~~ ")
    print('concordanceGlobal')
    numerateur = 0.0
    denom = 0.0
    # Pour chaque critère j
    #print('poids',poids)
    #print('len(h)-1 = ',len(h))
    for j in range(1,len(h)):
        print("critère ",j)
        numerateur += poids[j-1]*concordance_partiel(type[j-1],h,bi,j) # len(poids) = len(h)-2
        #TODO : si c'est correct, passer tout en list comprehension pour lisibilite
        print('numerateur',numerateur)
        denom += poids[j-1]
        print('denom', denom)
        #print('----------------')
        print('Ctemp(h,bi)',numerateur/denom)
        print("")
    print('C(h,bi)',numerateur/denom)
    print(" ~~~~~~~ ")
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
    print("h ",h)
    print("bi ",bi)
    print("Question : h surclasse t'il bi ?")
    if concordanceGlobal(h,bi,poids,type) >= seuilMajorite:
        print('Oui, H surclasse bi')
        print("")
        return True
    else:
        print('Non, H ne surclasse pas bi')
        print("")
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
    profil = len(classement)-1 # OPTIMISATION : commencer à len -2 car len-1 est forcement surclassé
    print('PROFIL ',profil)
    #print('H sur profil :', Surclasse(seuil, h, classement[profil], poids, type))
    #print('profil sur H : ', Surclasse(seuil, classement[profil], h, poids, type))
    print("###############")
    while (not(Surclasse(seuil, classement[profil], h, poids, type)) or (Surclasse(seuil, h, classement[profil], poids, type))): # anciennement 'and'
        print("Bilan : On passe au profil supérieur")
        print("###############")
        profil -= 1
    return profil #ou profil profil+1 ?


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
    liste = []
    dict = {}
    # pour chaque element de la matrice (une couche culotte), en partant de l'indice max (matrice profils[i][0]), on....
    print("Les couches", lesCouches)
    for couche in range(0,len(lesCouches)): # len(lesCouches)+1 ?
        # liste.append(
        #     ["Couche " + str(lesCouches[couche][0]),classement[AffectationOptimiste(lesCouches[couche],classement,poids,type,seuil)][0]]
        # )
        #print("Couche "+str(lesCouches[couche][0])+" :",classement[AffectationOptimiste(lesCouches[couche],classement,poids,type,seuil)])
        dict["Couche " + str(lesCouches[couche][0])] = categorie[AffectationOptimiste(lesCouches[couche], classement, poids, type, seuil)]
    #return liste
    return dict

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
    dict = {}
    for couche in range(0,len(lesCouches)): # len(lesCouches)+1 ?
        # liste.append(
        #     ["Couche " + str(lesCouches[couche][0]),classement[AffectationPessimiste(lesCouches[couche], classement, poids, type, seuil)][0]]
        # )
        #print("Couche " + str(lesCouches[couche][0]) + " :",classement[AffectationPessimiste(lesCouches[couche], classement, poids, type, seuil)])
        dict["Couche " + str(lesCouches[couche][0])] = categorie[AffectationPessimiste(lesCouches[couche], classement, poids, type, seuil)-1]   #return liste
    #return liste
    return dict

##
# main
##
import pandas as pd
import csv

poids = [3/5,2/5]
types = ['max','max']
SEUIL = 0.55
mat = []
matrice_profils = [
    ['Profil 6 : Frontiere Le meilleur, impossible',100,100],
    ['Profil 5 : Dans les premiers',3,3],
    ['Profil 4 : Moyen +',2,2],
    ['Profil 3 : Moyen -',1,1],
    ['Profil 2 : Dans les derniers',-1,-1],
    ['Profil 1 : Frontiere Le pire,impossible',-100,-100]
]

categorie = ['Très bon','Bon','Acceptable','Insuffisant','Inacceptable']

mat[5:6]
with open('mat.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    # on parse le fichier, en précisant les types pour chaque colonne
    mat = [[row[0], int(row[1]), int(row[2])] for row in reader]

# print("Données des couches mat : ",mat)
# print("Les profils :",matrice_profils)
# print("Les poids :",poids)
# print("Le type de notation",types)
mat[9:10]
print("#### DEBUT ELECTRE TRI ###")
print("optimiste")
op = EvalOptimiste(mat,matrice_profils,poids,types,SEUIL)
# test Buyin
# op = EvalOptimiste(mat,matrice_profils,poids,types,SEUIL)

op

print("pessimiste")
pe = EvalPessimiste(mat,matrice_profils,poids,types,SEUIL)
pe
# merging result
final_results = {key:[op[key], pe[key]] for key in op}
final_results
df = pd.DataFrame(final_results)
df = pd.DataFrame.from_records(final_results).T
df.columns = ['Electre Optimiste','Electre Pessimiste']
