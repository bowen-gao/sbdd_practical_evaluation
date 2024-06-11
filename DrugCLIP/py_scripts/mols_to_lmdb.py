import os

import rdkit

from rdkit import Chem

from rdkit.Chem import AllChem

import lmdb

import pickle

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

path = "/drug/sbdd_bench/ligan_pdbbind_0.9"

path = "/drug/sbdd_bench/ligan_pcba"







for file in os.listdir(path):
    if not file.endswith(".sdf"):
        continue
    target = file.split(".")[0]

    #target = target.split("_")[1]

    # read sdf file

    try:
        suppl = Chem.SDMolSupplier(f"{path}/{file}")
    except:
        continue

    data = []

    for mol in suppl:
        if mol is None:
            continue
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        #print(atom_types)
        coordinates = mol.GetConformer().GetPositions()

        #print(len(coordinates))
        coordinates = [coordinates]

        d = {
            "atoms": atoms,
            "coordinates": coordinates,
        }
        data.append(d)

    
    os.makedirs(f"{path}/mols", exist_ok=True)
    write_lmdb(data, f"{path}/mols/{target}.lmdb")



