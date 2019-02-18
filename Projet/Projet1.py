import csv
import numpy as np

def readFileCSVData(file):
    """
    Lit un fichier CSV
    :param file:
    :return: ponderation : list
             data : list
             result : list
    """
    ponderation=[]
    data=[]
    result=[]#derniere colonne
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if row[0]=='': #lecture des ponderations
                for i in range(1,len(row)-1):
                    ponderation.append(row[i])
            else:#lecture des donnees et des resultats
            #donnees contient les notes et result la moyenne
                temp=[]
                for i in range(1,len(row)-1):
                    temp.append(row[i])
                dicodata={row[0]:temp}
                dicoresult={row[0]:row[len(row)-1]}
                data.append(dicodata)
                result.append(dicoresult)
                
    return ponderation, data, result   

ponderation, data, result = readFileCSVData('criteres.csv')

def readFileCSVInt(file):
    intervalle=[]
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            dico={row[0]:[row[1],row[2]]}
            intervalle.append(dico)
    return intervalle
    
intervalle = readFileCSVInt('intervalles.csv')
print(len(intervalle))

