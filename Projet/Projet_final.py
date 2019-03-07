import csv
import numpy as np
from optlang import Model, Variable, Constraint, Objective

##

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


##
def readFileCSVInterval(file):
    intervalle={}
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            intervalle[row[0]]=[float(row[1]),float(row[2])]
    return intervalle



## Question 2.1 et 2.2
def createModel2(ponderation, data, result, intervalle,fixed):
    """
    fixed = True ===> fixer les variables f (question 2.1)
    fixed = False ===> ne pas fixer les variables f (question 2.2)
    """
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
    
    #creation des contraintes (1) a (15)
    C=[]
    for i in range(nbelements):
        calcul=0
        for j in range(nbcriteres):
            calcul+=ponderation[j]*U[i][j]
        c=Constraint(calcul-f[i],ub=0,lb=0)
        C.append(c)
    
    #creation des contraintes (16) a (19)
    
    for i in range(len(result)-1):
        if result[i][1]==result[i+1][1]:
            c=Constraint(f[i]-f[i+1],ub=0,lb=0)
            C.append(c)
        else:
            c=Constraint(f[i]-f[i+1]-0.1,lb=0)
            C.append(c)

    #creation des contraintes (20) a (29)
    for i in range(nbelements):
        for j in range(nbcriteres):
            c=Constraint(U[i][j],ub=intervalle.get(data[i][1][j])[1],lb=intervalle.get(data[i][1][j])[0])
            C.append(c)

    #Fixation des variables f 
    if fixed:
        for i in range(nbelements):
            c = Constraint(f[i],lb=result[i][1],ub=result[i][1])
            C.append(c)
    #
    
    obj = Objective(f[0], direction='max')
    model = Model(name='Modele')
    model.objective = obj
    model.add(C)
    return model
    
## Question 3

def createModel3(ponderation, data, result, intervalle):
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
    
    #creation des contraintes (1) a (15)
    C=[]
    for i in range(nbelements):
        calcul=0
        for j in range(nbcriteres):
            calcul+=ponderation[j]*U[i][j]
        c=Constraint(calcul-f[i],ub=0,lb=0)
        C.append(c)
    
    #creation des contraintes (16) a (19)
    for i in range(len(result)-1):
        
        if i==3 or i==8:
            continue
            
        elif result[i][1]==result[i+1][1]:
            c=Constraint(f[i]-f[i+1],ub=0,lb=0)
            C.append(c)
        else:
            c=Constraint(f[i]-f[i+1]-0.1,lb=0)
            C.append(c)

    
    #creation des contraintes (20) a (29)
    for i in range(nbelements):
        for j in range(nbcriteres):
            if i == 1 and j == 2:
                c=Constraint(U[i][j],ub=intervalle.get(data[i][1][j])[1])
                C.append(c)
            else:
                c=Constraint(U[i][j],ub=intervalle.get(data[i][1][j])[1],lb=intervalle.get(data[i][1][j])[0])
                C.append(c)
    
    obj = Objective(f[0], direction='max')
    model = Model(name='Modele')
    model.objective = obj
    
    model.add(C)
    return model
    
## Question 4
def createModel4(ponderation, data, result, intervalle):
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

    #creation des contraintes (1) a (15)
    C=[]
    for i in range(nbelements):
        calcul=0
        for j in range(nbcriteres):
            calcul+=ponderation[j]*U[i][j]
        c=Constraint(calcul-f[i],ub=0,lb=0)
        C.append(c)
 
    
    #creation des contraintes (20) a (29)
    for i in range(nbelements):
        for j in range(nbcriteres):
            c=Constraint(U[i][j],ub=intervalle.get(data[i][1][j])[1],lb=intervalle.get(data[i][1][j])[0])
            C.append(c)
    
    obj = Objective(f[0], direction='max')
    model = Model(name='Modele')
    model.objective = obj
    #for u in U:
        #model.add(u)
    #model.add(f)
    model.add(C)
    return model    
    
##
    
def CheckAdditiveModel(fileData, fileInt,question):
    ponderation, data, result = readFileCSVData(fileData)
    intervalle = readFileCSVInterval(fileInt)
    if question == "2.1":
        model = createModel2(ponderation, data, result, intervalle,False)
    elif question == "2.2":
        model = createModel2(ponderation, data, result, intervalle,True)
    elif question == "3":
        model = createModel3(ponderation, data, result, intervalle)
    elif question == "4":
        model = createModel4(ponderation, data, result, intervalle)
    else: 
        print("Existe pas") 
        return
    
    status = model.optimize()
    print("status:", model.status)
    print("objective value:", model.objective.value)
    print("----------")
    for var_name, var in model.variables.items():
        if var_name.startswith("f"):
            print(var_name, "=", var.primal)

##

def main():
#    
    print("----------------------------------------")
    print("Question 2.1")
    CheckAdditiveModel('criteres.csv', 'intervalles.csv',"2.1")
    print("----------------------------------------")
    print("Question 2.2")
    CheckAdditiveModel('criteres.csv', 'intervalles.csv',"2.2")
    print("----------------------------------------")
    print("Question 3")
    CheckAdditiveModel('criteres.csv', 'intervalles.csv',"3")
    print("----------------------------------------")
    print("Question 4")
    CheckAdditiveModel('criteres.csv', 'intervalles.csv',"4")
    print("----------------------------------------")
    print("test")
    CheckAdditiveModel('criteres.csv', 'intervalles.csv',"2")
    
#
    print("")
    print("AEROSOL")
    print("----------------------------------------")
    print("Question 2.1")
    CheckAdditiveModel('criteresAEROSOL.csv', 'intervalles.csv',"2.1")
    print("----------------------------------------")
    print("Question 2.2")
    CheckAdditiveModel('criteresAEROSOL.csv', 'intervalles.csv',"2.2")
    print("----------------------------------------")
    print("Question 3")
    CheckAdditiveModel('criteresAEROSOL.csv', 'intervalles.csv',"3")
    print("----------------------------------------")
    print("Question 4")
    CheckAdditiveModel('criteresAEROSOL.csv', 'intervalles.csv',"4")
    print("----------------------------------------")


if __name__ == "__main__":
    main()




















