from .graph_conv_many_nuc_pred import Model
from rdkit.Chem import AllChem
from rdkit import Chem


class NMRPredictor:
    def __init__(self, modelMeta, modelState, USE_CUDA = True):
        self.model = Model(modelMeta, modelState, USE_CUDA)

    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        decoyVal = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                decoyVal[atom.GetIdx()] = 0
        
        decoyVal = [decoyVal]
        if mol.GetNumConformers() <= 0:
            raise "Coordinates didn't converge"

        df = self.model.pred([mol], [decoyVal])

        return df.est.tolist()        
