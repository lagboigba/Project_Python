import csv


def readFileCSV(file):
    """
    Lit un fichier CSV
    :param file:
    :return: ponderation : list
             data : list
             result : list
    """
    ponderation=[]
    data=[]
    result=[]
    data.append('truc')
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if row[0]=='':
                for i in range(1,len(row)-1):
                    ponderation.append(row[i])
            else:
                print ('Haram')
                
    return ponderation, data, result   


ponderation, data, result = readFileCSV('criteres.csv')
print(ponderation)
print(data)