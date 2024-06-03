

import os

import rdkit

from rdkit import Chem

from rdkit.Chem import AllChem

from tqdm import tqdm

import lmdb

import pickle

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    out_list = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        
        data = pickle.loads(datapoint_pickled)
        out_list.append(data)
    return out_list

def write_lmdb(out_list, save_path):
    
    env = lmdb.open(
        save_path,
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776
    )

    with env.begin(write=True) as lmdb_txn:
        for i in range(len(out_list)):
            #print('{:0>10d}'.format(i), pickle.dumps(out_list[i]))
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))

with open("FDA_Approved_structures.csv", "r") as f:
    lines = f.readlines()


data_list = []

# for i in tqdm(range(1, len(lines))):
#     lis = lines[i].strip().split(",")
#     #print(lis)
#     if len(lis)<2:
#         continue
    
#     smi = lis[-1]

#     try:
#         mol = Chem.MolFromSmiles(smi)


#         # generated conformation

#         mol = Chem.AddHs(mol)

#         AllChem.EmbedMolecule(mol)

#         AllChem.MMFFOptimizeMolecule(mol)

#         mol = Chem.RemoveHs(mol)
    
    
#         atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

#         data = {
#             "atoms": atom_symbols,
#             "coordinates": [mol.GetConformer().GetPositions().tolist()],
#             "smiles": smi
#         }

#         data_list.append(data)
#     except:
#         print("Error in smiles: ", smi)
#         continue


#write_lmdb(data_list, "FDA_Approved.lmdb")

data = read_lmdb("FDA_Approved.lmdb")
import numpy as np

for d in data:
    d["coordinates"] = np.array(d["coordinates"])




write_lmdb(data, "FDA_Approved_new.lmdb")




    