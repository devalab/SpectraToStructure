import os
import rdkit
from rdkit import Chem
from rdkit import DataStructs
direc = "./test/"
files = os.listdir(direc)
import pickle
total = 0
correct = 0
def customKey(item):
    return (len(item), item)

for file in sorted(files, key=customKey):
    if file[:5] != "smile":
        continue
    f = open(direc+file,"rb")
    liDict = pickle.load(f)
    total += 1
    maxRes = 0
    bestMolSmiles = "EMPTY"
    targetSmiles = list(liDict[0].keys())[0]
    targetFP = Chem.RDKFingerprint(Chem.MolFromSmiles(targetSmiles))
    found = False
    for data in liDict:
        if 1 in data[list(data.keys())[0]]:
            correct += 1
            print("T: {:20}, B: {:20}, Similarity: {:.3f}, FileName: {}".format(targetSmiles, targetSmiles, 1, file))
            found = True
            break
            # print(data)
            # print(file)
        for pred in data[list(data.keys())[0]]: 
            if not isinstance(pred, str):
                continue
            predFP = Chem.RDKFingerprint(Chem.MolFromSmiles(pred))
            sim = DataStructs.FingerprintSimilarity(targetFP, predFP)
            if sim >= maxRes:
                maxRes = sim
                bestMolSmiles = pred

    if found:
        continue

    print("T: {:20}, B: {:20}, Similarity: {:.3f}, FileName: {}".format(targetSmiles, bestMolSmiles, maxRes, file))

print(correct/total)
print("Total:" , total)
