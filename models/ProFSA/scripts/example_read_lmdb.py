import pickle as pkl

import lmdb

path = "/data/prot_frag/pocket_matching/test_ligand.lmdb"
env = lmdb.open(
    path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)

with env.begin(write=False) as txn:
    keys = list(txn.cursor().iternext(values=False))

for key in keys:
    print(key)
    with env.begin(write=False) as txn:
        value = txn.get(key)
    data = pkl.loads(value)
    print(data)
    break
