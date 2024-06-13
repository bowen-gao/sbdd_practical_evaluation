import pickle

import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import random

import lmdb
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem.AllChem as AllChem
import torch
from tqdm import tqdm
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


def write_lmdb(data, lmdb_path):
    #resume

    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    num = 0
    with env.begin(write=True) as txn:
        for d in tqdm(data):
            txn.put(str(num).encode('ascii'), pickle.dumps(d))
            num += 1

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
    i=0
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        out_list.append(data)
    env.close()
    return out_list 
