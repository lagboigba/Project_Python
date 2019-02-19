import csv
import numpy as np
from optlang import Model, Variable, Constraint, Objective

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
                    ponderation.append(float(row[i]))
            else:#lecture des donnees et des resultats
            #donnees contient les notes et result la moyenne
                temp=[]
                for i in range(1,len(row)-1):
                    temp.append(row[i])
                #dicodata={row[0]:temp}
                #dicoresult={row[0]:float(row[len(row)-1])}
                data.append([row[0],temp])
                result.append([row[0],float(row[len(row)-1])])
                
    return ponderation, data, result   

ponderation, data, result = readFileCSVData('criteres.csv')

def readFileCSVInt(file):
    intervalle={}
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            #dico={row[0]:[row[1],row[2]]}
            intervalle[row[0]]=[float(row[1]),float(row[2])]
    return intervalle
    
intervalle = readFileCSVInt('intervalles.csv')

def createModel(ponderation, data, result, intervalle):
    nbelements=len(data)
    nbcriteres=len(ponderation)
    #creation des variables
    U=[]
    f=[]
    for i in range(nbelements):
        x=Variable("f"+str(i+1),lb=0)
        f.append(x)
        x=[]
        for j in range(nbcriteres):
            x1=Variable("U"+str(i+1)+str(j+1),lb=0)
            x.append(x1)
        U.append(x)
    print(U)
    #print(f)
    
    #creation des contraintes (1) a (15)
    C=[]
    for i in range(nbelements):
        calcul=0
        for j in range(nbcriteres):
            calcul+=ponderation[j]*U[i][j]
        c=Constraint(calcul-f[i],ub=0,lb=0)
        C.append(c)
    #print(C[0].to_json())
    
##changement question 4
    #creation des contraintes (16) a (19)
    #print("C :",len(C))
    #for i in range(len(result)-1):
        ##changement question 3
        #if i==3 or i==8:
            #continue
        ##
        #elif result[i][1]==result[i+1][1]:
           #c=Constraint(f[i]-f[i+1],ub=0,lb=0)
            #C.append(c)
        #else:
           # c=Constraint(f[i]-f[i+1]-0.1,lb=0)
            #C.append(c)
##
    #print("C :",len(C))
    
    #creation des contraintes (20) a (29)
    for i in range(nbelements):
        for j in range(nbcriteres):
            ##changement question 3
            if i == 1 and j == 2:
                c=Constraint(U[i][j],ub=intervalle.get(data[i][1][j])[1])
                C.append(c)
            ##
            else:
                c=Constraint(U[i][j],ub=intervalle.get(data[i][1][j])[1],lb=intervalle.get(data[i][1][j])[0])
                C.append(c)
    #print("result :", result)
    
    ##Fixation des variables f : à enlever pour ne pas les fixer
    #for i in range(nbelements):
        #c = Constraint(f[i],lb=result[i][1],ub=result[i][1])
        #C.append(c)
    ##
    
    obj = Objective(f[11], direction='max')
    model = Model(name='Modele')
    model.objective = obj
    #for u in U:
        #model.add(u)
    #model.add(f)
    model.add(C)
    return model

def CheckAdditiveModel(fileData, fileInt):
    ponderation, data, result = readFileCSVData(fileData)
    intervalle = readFileCSVInt(fileInt)
    model = createModel(ponderation, data, result, intervalle)
    status = model.optimize()
    print("status:", model.status)
    print("objective value:", model.objective.value)
    print("----------")
    for var_name, var in model.variables.items():
        if var_name.startswith("f"):
            print(var_name, "=", var.primal)

    
#model = createModel(ponderation, data, result, intervalle)
#print(model)
CheckAdditiveModel('criteres.csv', 'intervalles.csv')