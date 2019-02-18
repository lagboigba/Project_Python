# Poids associés aux critères
W = [3/5,2/5]
print(W)

matrice_profils = [
    [4,4],
    [3,3],
    [2,2],
    [1,1],
    [-1,-1],
    [-3,-3]
]
couches = ['Profil 6',
           'Profil 5',
           'Profil 4',
           'Profil 3',
           'Profil 2',
           'Profil 1']

# Calcul cj

def concordance_partiel(type,h,bi,j):
    """

    :param h: Couche-culotte
    :param bi: profil (frontière qui YYYY)
    :return: 1 si H est au-moins aussi bon que bi, 0 sinon
    """
    if(type == 'max'):
        if h[j] >= matrice_profils[bi][j]:
            return 1
        else :
            return 0
    elif(type=='min'):
        if h[j] <= matrice_profils[bi][j]:
            return 1
        else :
            return 0
    else:
        print("Calcul concordance partielle impossible car type inconnu : ", type)


##
# main
##
import csv
mat = []
with open('mat.csv', 'r') as f:
    reader = csv.reader(f)
    row = list(reader)
    mat.append(row)
#concordance_partiel('max',)

print (mat)