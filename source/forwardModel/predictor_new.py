from .graph_conv_many_nuc_pred import Model
from rdkit.Chem import AllChem
from rdkit import Chem
import pickle
import copy
import torch
class NMRPredictor:
    def __init__(self, modelMeta, modelState, USE_CUDA = True):

        if not torch.cuda.is_available():
            raise Exception("GPU not available here :(")
        

        self.model = Model(modelMeta, modelState, USE_CUDA)
        self.cache = None
        try:
            with open("../../data/PredictorCache.pkl", "rb") as inFile:
                self.cache = pickle.load(inFile)
            
            print("Loaded Cache of predictor with size, ", len(self.cache))
        except: 
            self.cache = None
        self.cacheNew = copy.copy(self.cache)

    def __del__(self):
        pass
        #self.save()

    def save(self):
        #print("Added " + str(len(self.cacheNew) - len(self.cache)) +" new mols in cache") 
        with open("../../data/PredictorCache.pkl", "wb") as outFile:
            pickle.dump(self.cacheNew, outFile)
        self.cache = copy.copy(self.cacheNew)

    def predict(self, smiles):
        if self.cacheNew and self.cacheNew.get(smiles):
            val = self.cacheNew.get(smiles)
            # print("USED CACHE VALUE ", val)
            return val

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        decoyVal = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                decoyVal[atom.GetIdx()] = 0
        
        decoyVal = [decoyVal]

        try:
            with torch.no_grad():
                df = self.model.pred([mol], [decoyVal])
        except Exception as e:
            if len(self.cacheNew) < 1000006:
                self.cacheNew[smiles] = -1
            return -1
        
        if self.cacheNew is None:
            self.cacheNew = dict()
            self.cache = dict()
        
        #print("Cache Sizes: ", len(self.cacheNew), len(self.cache))
        if len(self.cacheNew) > len(self.cache) + 1000 :
            self.save()
        if len(self.cacheNew) < 1000006:
            self.cacheNew[smiles] = df.est.tolist()
        return df.est.tolist()        
