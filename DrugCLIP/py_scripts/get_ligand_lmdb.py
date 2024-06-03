import os
import pickle
import lmdb
import selfies as sf
from tqdm import tqdm, trange

from rdkit import Chem

from rdkit.Chem import AllChem
import numpy as np



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
        if len(data["pocket_coordinates"])==0:
            continue
        out_list.append(data)

        


    env.close()
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
        for i in tqdm(range(len(out_list))):
            #print('{:0>10d}'.format(i), pickle.dumps(out_list[i]))
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))




root_path = "/data/protein/DUD-E/raw/all/"

root_path = "/data/protein/lib-pcba/raw/lit_pcba/"

targets = os.listdir(root_path)

for target in targets:

    data_list = []
    if os.path.exists(os.path.join(root_path, target, "ligand.lmdb")):
        os.remove(os.path.join(root_path, target, "ligand.lmdb"))

    for file in os.listdir(os.path.join(root_path, target)):
       
        if file.endswith("_ligand.mol2"):
            ligand_path = os.path.join(root_path, target, file)

            lig_mol = Chem.MolFromMol2File(ligand_path, sanitize=False)
            if lig_mol is None:
                continue
            #smi = Chem.MolToSmiles(lig_mol)
            # remove Hs
            #lig_mol = Chem.RemoveHs(lig_mol)
            # atom names
            atoms = [lig_mol.GetAtomWithIdx(i).GetSymbol() for i in range(lig_mol.GetNumAtoms())]
            
            coordinates = np.array(lig_mol.GetConformer().GetPositions())
            assert len(atoms) == coordinates.shape[0]

            ligand = {'atoms': atoms, 'coordinates': coordinates, 'smi': "1", "label": 1}

            data_list.append(ligand)
            

    #ligand_path = os.path.join(root_path, target, "crystal_ligand.mol2")

    #data_list.append(ligand)


    write_lmdb(data_list, os.path.join(root_path, target, "ligand.lmdb"))
    
   