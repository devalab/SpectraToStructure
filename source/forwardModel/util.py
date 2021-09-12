import contextlib
import os
import numpy as np
import tempfile
#import cirpy

from sklearn.cluster import AffinityPropagation

from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
#import pubchempy as pcp
import rdkit
import math
import sklearn.metrics.pairwise
import scipy.optimize
import pandas as pd
import re 
import itertools
import time
#import numba
import torch

from . import nets
from tqdm import tqdm

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


CHEMICAL_SUFFIXES = ['ane', 'onl', 'orm', 'ene', 'ide', 'hyde', 'ile', 'nol', 
                     'one', 'ate', 'yne', 'ran', 'her', 'ral', 'ole', 'ine']

@contextlib.contextmanager
def cd(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)


def conformers_best_rms(mol):
   num_conformers = mol.GetNumConformers()
   best_rms = np.zeros((num_conformers, num_conformers))
   for i in range(num_conformers):
       for j in range(num_conformers):
           best_rms[i, j] = AllChem.GetBestRMS(mol, mol, prbId=i, refId=j)
   return best_rms

def cluster_conformers(mol):
   """
   return the conformer IDs that represent cluster centers
   using affinity propagation 

   return conformer positions from largest to smallest cluster

   """
   best_rms = conformers_best_rms(mol)

   af = AffinityPropagation(affinity='precomputed').fit(best_rms)
   cluster_centers_indices = af.cluster_centers_indices_
   labels = af.labels_
   n_clusters_ = len(cluster_centers_indices)
   
   cluster_sizes = np.zeros(n_clusters_)
   for i in range(n_clusters_):
      cluster_sizes[i]= np.sum(labels == i)
   sorted_indices = cluster_centers_indices[np.argsort(cluster_sizes)[::-1]]

   return sorted_indices, labels, best_rms

def GetCalcShiftsLabels(numDS, BShieldings, labels, omits, 
                        TMS_SC_C13 = 191.69255,
                        TMS_SC_H1 = 31.7518583):
    """
    originally from pydp4
    """
     
    Clabels = []
    Hlabels = []
    Cvalues = []
    Hvalues = []
    
    for DS in range(numDS):

        Cvalues.append([])
        Hvalues.append([])

        #loops through particular output and collects shielding constants
        #and calculates shifts relative to TMS
        for atom in range(len(BShieldings[DS])):
            shift = 0
            atom_label = labels[atom]
            atom_symbol = re.match("(\D+)\d+", atom_label).groups()[0]

            if atom_symbol == 'C' and not labels[atom] in omits:
                # only read labels once, i.e. the first diastereomer
                if DS == 0:
                    Clabels.append(labels[atom])
                shift = (TMS_SC_C13-BShieldings[DS][atom]) / \
                    (1-(TMS_SC_C13/10**6))
                Cvalues[DS].append(shift)

            if atom_symbol == 'H' and not labels[atom] in omits:
                # only read labels once, i.e. the first diastereomer
                if DS == 0:
                    Hlabels.append(labels[atom])
                shift = (TMS_SC_H1-BShieldings[DS][atom]) / \
                    (1-(TMS_SC_H1/10**6))
                Hvalues[DS].append(shift)

    return Cvalues, Hvalues, Clabels, Hlabels


def mol_to_sdfstr(mol):
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as fid:

       writer = Chem.SDWriter(fid)
       writer.write(mol)
       writer.close()
       fid.flush()
       fid.seek(0)
       return fid.read()

def download_cas_to_mol(molecule_cas, sanitize = True):
    """
    Download molecule via cas, add hydrogens, clean up
    """
    sdf_str = cirpy.resolve(molecule_cas, 'sdf3000', get_3d=True)
    mol = sdbs_util.sdfstr_to_mol(sdf_str)
    mol = Chem.AddHs(mol)

    # this is not a good place to do this
    # # FOR INSANE REASONS I DONT UNDERSTAND we get
    # #  INITROT  --  Rotation about     1     4 occurs more than once in Z-matrix
    # # and supposeldy reordering helps

    # np.random.seed(0)
    # mol = Chem.RenumberAtoms(mol, np.random.permutation(mol.GetNumAtoms()).astype(int).tolist())
    
    #mol.SetProp("_Name", molecule_cas)
    # rough geometry 
    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    return mol

def check_prop_failure(is_success, infile, outfile):
    if not is_success:
        pickle.dump({"success" : False, 
                     "previous_success" : False, 
                     'infile' : infile},
                    open(outfile, 'wb'))
    return not is_success
        

def pubchem_cid_to_sdf(cid, cleanup_3d=True):
    """
    Go from pubmed CID to 
    """
    with tempfile.TemporaryDirectory() as tempdir:
        fname = f'{tempdir}/test.sdf'
        #pcp.download('SDF',fname , cid, 'cid', overwrite=True)
        raise "PCP Needed`"
        suppl = Chem.SDMolSupplier(fname, sanitize=True)
        mol = suppl[0]
        mol = Chem.AddHs(mol)
        if cleanup_3d:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        return mol
    
def render_2d(mol):
    mol = Chem.Mol(mol)
    
    AllChem.Compute2DCoords(mol)
    return mol


def array_to_conf(mat):
    """
    Take in a (N, 3) matrix of 3d positions and create
    a conformer for those positions. 
    
    ASSUMES atom_i = row i so make sure the 
    atoms in the molecule are the right order!
    
    """
    N = mat.shape[0]
    conf = Chem.Conformer(N)
    
    for ri in range(N):
        p = rdkit.Geometry.rdGeometry.Point3D(*mat[ri])                                      
        conf.SetAtomPosition(ri, p)
    return conf



def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    
    From https://stackoverflow.com/a/6802723/1073963
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotate_mat(theta, phi):
    """
    generate a rotation matrix with theta around x-axis
    and phi around y
    """
    return np.dot(rotation_matrix([1, 0, 0], theta), 
                  rotation_matrix([0, 1, 0], phi),)

def mismatch_dist_mat(num_a, num_b, mismatch_val=100):
    """
    Distance handicap matrix. Basically when matching elements
    from num_a to num_b, if they disagree (and thus shoudln't be
    matched) add mismatch_value
    """
    
    m = np.zeros((len(num_a), len(num_b)))
    for i, a_val in enumerate(num_a):
        for j, b_val in enumerate(num_b):
            if a_val != b_val:
                m[i, j] = mismatch_val
    return m

def create_rot_mats(ANGLE_GRID_N = 48):
    """
    Create a set of rotation matrices through angle gridding
    """
    
    theta_points = np.linspace(0, np.pi*2, ANGLE_GRID_N, endpoint=False)
    rotate_points = np.array([a.flatten() for a in np.meshgrid(theta_points, theta_points)]).T
    rot_mats = np.array([rotate_mat(*a) for a in rotate_points])

    return rot_mats

def weight_heavyatom_mat(num_a, num_b, heavy_weight = 10.0):
    """

    """
    
    m = np.zeros((len(num_a), len(num_b)))
    for i, a_val in enumerate(num_a):
        for j, b_val in enumerate(num_b):
            if a_val > 1 and b_val > 1:
                m[i, j] = heavy_weight
    return m

def compute_rots_and_assignments(points_1, points_2, dist_mat_mod = None, ANGLE_GRID_N = 48):
    """
    Compute the distance between points for all possible
    gridded rotations. 
    
    """
    
    rot_mats = create_rot_mats(ANGLE_GRID_N)

    all_test_points = np.dot(rot_mats, points_2.T)
    
    
    
    total_dists = []
    assignments = []
    for test_points in all_test_points:

        dist_mat = sklearn.metrics.pairwise.euclidean_distances(points_1, 
                                                                test_points.T)
        if dist_mat_mod is not None:
            dist_mat += dist_mat_mod
        cost_assignment = scipy.optimize.linear_sum_assignment(dist_mat)
        assignments.append(cost_assignment)

        match_distances =dist_mat[np.array(list(zip(*cost_assignment)))]
        total_dist = np.sum(match_distances)

        total_dists.append(total_dist)
    assert(assignments[0][0].shape[0] == points_1.shape[0])
    return total_dists, assignments


def find_best_ordering(sdf_positions, sdf_nums, 
                      table_positions, table_nums, mismatch_val=100):
    """
    Find the ordering of table_positions that minimizes
    the distance between it and sdf_positions at some rotation
    """
    mod_dist_mat = mismatch_dist_mat(sdf_nums, table_nums, mismatch_val=mismatch_val)
    mod_dist_mat += weight_heavyatom_mat(sdf_nums, table_nums, 10.0)
    #print(mod_dist_mat)
    total_dists, assignments = compute_rots_and_assignments(sdf_positions, table_positions, 
                                                                dist_mat_mod=mod_dist_mat, 
                                                                ANGLE_GRID_N = 48)
  
    best_assign_i = np.argmin(total_dists)
    #pylab.axvline(best_assign_i, c='r')
    best_assignment = assignments[best_assign_i]
    return best_assignment[1], total_dists[best_assign_i]


def explode_df(df, lst_cols, fill_value=''):
    """
    Take a data frame with a column that's a list of entries and return
    one with a row for each element in the list
    
    From https://stackoverflow.com/a/40449726/1073963
    
    """
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
            


def generate_canonical_fold_sets(BLOCK_N, HOLDOUT_N):
    """
    This generates a canonical ordering of N choose K where:
    1. the returned subset elements are always sorted in ascending order
    2. the union of the first few is the full set

    This is useful for creating canonical cross-validation/holdout sets
    where you want to compare across different experimental setups
    but you want to make sure you see all the data in the first N
    """


    if BLOCK_N % HOLDOUT_N == 0:
        COMPLETE_FOLD_N = BLOCK_N // HOLDOUT_N
        # evenly divides, we can do sane thing
        init_sets = []
        for i in range(HOLDOUT_N):
            s = np.array(np.split((np.arange(BLOCK_N) + i) % BLOCK_N, COMPLETE_FOLD_N))
            init_sets.append([sorted(i) for i in s])
        init_folds = np.concatenate(init_sets)

        all_folds = set([tuple(sorted(a)) for a in itertools.combinations(np.arange(BLOCK_N), 
                                                                  HOLDOUT_N)])

        # construct set of init
        init_folds_set =  set([tuple(a) for a in init_folds])
        assert len(init_folds_set) == len(init_folds)
        assert init_folds_set.issubset(all_folds)
        non_init_folds = all_folds - init_folds_set

        all_folds_array = np.zeros((len(all_folds), HOLDOUT_N), dtype=np.int)
        all_folds_array[:len(init_folds)] = init_folds
        all_folds_array[len(init_folds):] = list(non_init_folds)

        return all_folds_array
    else:
        raise NotImplementedError()

def dict_product(d):
    dicts = {}
    for k, v in d.items():
        if not isinstance(v, (list, tuple, np.ndarray)):
            v = [v]
        dicts[k] = v
        
    return list((dict(zip(dicts, x)) for x in itertools.product(*dicts.values())))


class SKLearnAdaptor(object):
    def __init__(self, model_class, 
                 feature_col, pred_col, 
                 model_args, save_debug=False):
       """
       feature_col is either : 
       1. a single string for a feature column which will be flattened and float32'd
       2. a list of [(df_field_name, out_field_name, dtype)]
       """

       self.model_class = model_class
       self.model_args = model_args

       self.m = self.create_model(model_class, model_args)
       self.feature_col = feature_col
       self.pred_col = pred_col
       self.save_debug = save_debug

    def create_model(self, model_class, model_args):
       return  model_class(**model_args)

    def get_X(self, df):
       if isinstance(self.feature_col, str):
          # do the default thing
          return np.vstack(df[self.feature_col].apply(lambda x: x.flatten()).values).astype(np.float32)
       else:
          # X is a dict of arrays
          return {out_field : np.stack(df[in_field].values).astype(dtype) \
                  for in_field, out_field, dtype in self.feature_col}

    def fit(self, df, partial=False):
        X = self.get_X(df)
        y = np.array(df[self.pred_col]).astype(np.float32).reshape(-1, 1)
        if isinstance(X, dict):
           for k, v in X.items():
              assert len(v) == len(y)
        else:
           assert len(X) == len(y)
        if self.save_debug:
           pickle.dump({'X' : X, 
                        'y' : y}, 
                       open("/tmp/SKLearnAdaptor.fit.{}.pickle".format(t), 'wb'), -1)
        if partial:
           self.m.partial_fit(X, y)
        else:
           self.m.fit(X, y)
        
    def predict(self, df):
        X_test = self.get_X(df)

        pred_vect = pd.DataFrame({'est' : self.m.predict(X_test).flatten()}, 
                                 index=df.index)
        if self.save_debug:
           pickle.dump({'X_test' : X_test, 
                        'pred_vect' : pred_vect}, 
                       open("/tmp/SKLearnAdaptor.predict.{}.pickle".format(t), 'wb'), -1)

        return pred_vect

#@numba.jit(nopython=True)
def create_masks(BATCH_N, row_types, out_types):
    MAT_N = row_types.shape[1]
    OUT_N = len(out_types)

    M = np.zeros((BATCH_N, MAT_N, MAT_N, OUT_N), 
                 dtype=np.float32)
    for bi in range(BATCH_N):
        for i in range(MAT_N):
            for j in range(MAT_N):
                for oi in range(OUT_N):
                    if out_types[oi] == row_types[bi, j]:
                        M[bi, i, j, oi] = 1
    return M

def numpy(x):
   """
   pytorch convenience method just to get a damn 
   numpy array back from a tensor or variable
   wherever the hell it lives
   """
   if isinstance(x, np.ndarray):
      return x
   if isinstance(x, list):
      return np.array(x)

   if isinstance(x, torch.Tensor):
      if x.is_cuda:
         return x.cpu().numpy()
      else:
         return x.numpy()
   raise NotImplementedError(str(type(x)))

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split_df(dfm, chunk_size):
   """
   For splitting a df in to chunks of approximate size chunk_size
   """
   indices = index_marks(dfm.shape[0], chunk_size)
   return np.array_split(dfm, indices)


def create_col_constraint(max_col_sum):
    """
    N = len(max_col_sum)
    for a NxN matrix x create a matrix A (N x NN*2)
    such that A(x.flatten)=b constrains the columns of x to equal max_col_sum
    
    return A, b
    """
    N = len(max_col_sum)
    A = np.zeros((N, N**2))
    b = max_col_sum
    Aidx = np.arange(N*N).reshape(N, N)
    for row_i, max_i in enumerate(max_col_sum):
        sub_i = Aidx[:, row_i]
        A[row_i, sub_i] = 1
    return A, b


def create_row_constraint(max_row_sum):
    """
    N = len(max_row_sum)
    for a NxN matrix x create a matrix A (N x NN*2)
    such that A(x.flatten)=b constrains the row of x to equal max_row_sum
    
    return A, b
    """
    N = len(max_row_sum)
    A = np.zeros((N, N**2))
    b = max_row_sum
    Aidx = np.arange(N*N).reshape(N, N)
    for row_i, max_i in enumerate(max_row_sum):
        sub_i = Aidx[row_i, :]
        A[row_i, sub_i] = 1
    return A, b

def row_col_sums(max_vals):
   
   Ac, bc = create_row_constraint(max_vals)

   Ar, br = create_col_constraint(max_vals)
   Aall = np.vstack([Ac, Ar])
   ball = np.concatenate([bc, br])
   return Aall, ball 


def adj_to_mol(adj_mat, atom_types):
   assert adj_mat.shape == (len(atom_types), len(atom_types))

   mol = Chem.RWMol()
   for atom_i, a in enumerate(atom_types):
       if a > 0:
           atom = Chem.Atom(int(a))
           idx = mol.AddAtom(atom)
   for a_i in range(len(atom_types)):
       for a_j in range(a_i + 1, len(atom_types)):
           bond_order = adj_mat[a_i, a_j]
           bond_order_int = np.round(bond_order)
           if bond_order_int == 0:
              pass
           elif bond_order_int == 1:
               bond = Chem.rdchem.BondType.SINGLE
           elif bond_order_int == 2:
               bond = Chem.rdchem.BondType.DOUBLE
           else: 
              raise ValueError()

           if bond_order_int > 0:
              mol.AddBond(a_i, a_j, order=bond)
   return mol

def get_bond_order(m, i, j):
   """
   return numerical bond order
   """
   b = m.GetBondBetweenAtoms(int(i), int(j))
   if b is None:
      return 0
   c = b.GetBondTypeAsDouble()
   return c


def get_bond_order_mat(m):
   """
   for a given molecule get the adj matrix with the right bond order
   """

   ATOM_N = m.GetNumAtoms()
   A = np.zeros((ATOM_N, ATOM_N))
   for i in range(ATOM_N):
       for j in range(i+1, ATOM_N):
         b = get_bond_order(m, i, j)
         A[i, j] = b
         A[j, i] = b
   return A


def get_bond_list(m):
    """
    return a multiplicty-respecting list of bonds
    """
    ATOM_N = m.GetNumAtoms()
    bond_list = []
    for i in range(ATOM_N):
        for j in range(i+1, ATOM_N):
            b = get_bond_order(m, i, j)
            for bi in range(int(b)):
                bond_list.append((i, j))
    return bond_list

def clear_bonds(mrw):
    """
    in-place clear bonds
    """
    ATOM_N = mrw.GetNumAtoms()
    for i in range(ATOM_N):
        for j in range(ATOM_N):
            if mrw.GetBondBetweenAtoms(i, j) is not None:
                mrw.RemoveBond(i, j)
    return mrw

def set_bonds_from_list(m, bond_list):
    """
    for molecule M, set its bonds from the list
    """
    mrw = Chem.RWMol(m)
    clear_bonds(mrw)
    for i, j in bond_list:
        b_order = get_bond_order(mrw, i, j)
        set_bond_order(mrw, i, j, b_order+1)
    return Chem.Mol(mrw)

def edge_array(G):
     return np.array(list(G.edges()))


def canonicalize_edge_array(X):
    """
    Sort an edge array first by making sure each
    edge is (a, b) with a <= b
    and then lexographically
    """
    Y = np.sort(X)
    return Y[np.lexsort(np.rot90(Y))]


def set_bond_order(m, i, j, order):
    i = int(i)
    j = int(j)
    # remove existing bond
    if m.GetBondBetweenAtoms(i, j) is not None:
        m.RemoveBond(i, j)
        
    order = int(np.floor(order))
    if order == 0:
        return
    if order == 1 :
        rd_order = rdkit.Chem.BondType.SINGLE
    elif order == 2:
        rd_order = rdkit.Chem.BondType.DOUBLE
    elif order == 3:
        rd_order = rdkit.Chem.BondType.TRIPLE
    else:
        raise ValueError(f"unkown order {order}")
        
    m.AddBond(i, j, order=rd_order)

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def conf_not_null(mol, conf_i):
    _, coords = get_nos_coords(mol, conf_i)
    
    if np.sum(coords**2) < 0.01:
        return False
    return True


def get_nos_coords(mol, conf_i):
    conformer = mol.GetConformers()[conf_i]
    coord_objs = [conformer.GetAtomPosition(i) for i in  range(mol.GetNumAtoms())]
    coords = np.array([(c.x, c.y, c.z) for c in coord_objs])
    atomic_nos = np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)
    return atomic_nos, coords

def get_nos(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)

def move(tensor, cuda=False):
    from torch import nn
    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()




def mol_df_to_neighbor_atoms(mol_df):
    """
    Take in a molecule df and return a dataframe mapping
    (mol_id, atom_idx) 
    """
    
    neighbors = []
    for mol_id, row in tqdm(mol_df.iterrows(), total=len(mol_df)):
        m = row.rdmol
        for atom_idx in range(m.GetNumAtoms()):
            a = m.GetAtomWithIdx(atom_idx)
            nas = a.GetNeighbors()
            r = {'mol_id' : mol_id, 'atom_idx' : atom_idx}
            for na in nas:
                s = na.GetSymbol()
                if s in r:
                    r[s] += 1
                else:
                    r[s] = 1
            r['num_atoms'] = m.GetNumAtoms()
            neighbors.append(r)
    neighbors_df = pd.DataFrame(neighbors).fillna(0).set_index(['mol_id', 'atom_idx'])
    return neighbors_df

