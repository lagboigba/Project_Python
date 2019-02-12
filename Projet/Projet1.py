import csv


def readFileCSV(file):
    ponderation=[]
    with open(file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if row[0]=='':
                for i in range(1,len(row)-1):
                    ponderation.append(row[i])
    return ponderation        
            
ponderation = readFileCSV('criteres.csv')
print(ponderation)