import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..protein_ligand import PDBProtein, parse_sdf_file
from ..data import ProteinLigandData, torchify_dict



class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, data_split_path, id_split_path, index_path, dataset_name,split_name, transform=None ):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.processed_path = os.path.join(index_path,
                                           dataset_name + f'.lmdb')
        self.index_path = os.path.join(index_path, dataset_name + '.pkl')
        print("self.index_path : ",self.index_path)
        print("self.processed_path : ",self.processed_path)
        self.transform = transform
        self.db = None

        self.keys = None

        # if lmdb is smaller than 100K, then remove it
        if os.path.exists(self.processed_path):
            if os.path.getsize(self.processed_path) < 100 * 1024:
                os.remove(self.processed_path)
                print(f'{self.processed_path} is too small, removed')
                if os.path.exists(self.processed_path + '-lock'):
                    os.remove(self.processed_path + '-lock')
                    print(f'{self.processed_path}-lock is removed')

        new_lmdb=False
        if not os.path.exists(self.processed_path) :
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
            new_lmdb=True
        
        if data_split_path is not None:
            self.data_split_file = os.path.join(data_split_path, split_name + '.pkl')
            self.id_split_file = os.path.join(id_split_path, split_name + '.pkl')
            if not os.path.exists(self.id_split_file) or new_lmdb:
                if new_lmdb:
                    print("new lmdb, refresh id_split_file")
                else :
                    print(f'{self.id_split_file} does not exist, begin processing id split file')
                self._process_id_split_file()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        num_data=0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    
                    # chech all bond valid
                    min_ligand_bond_type = min(ligand_dict['bond_type'])
                    if min_ligand_bond_type == 0:
                        raise ValueError("Invalid bond type in ligand")

                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )

                    # check2 
                    if not data.protein_pos.size(0) > 0:
                        raise ValueError("Empty protein position")
                    
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue

                    # check ligand_nbh_list
                    length=len(data["ligand_nbh_list"])
                    for i in range(length):
                        if i not in data["ligand_nbh_list"]:
                            print("ligand_nbh_list error")
                            print(data["ligand_nbh_list"])
                            print("ligand_fn : ",ligand_fn)
                            raise ValueError("ligand_nbh_list error")

                    txn.put(
                        key=f'{num_data:08d}'.encode(),
                        value=pickle.dumps(data)
                    )
                    num_data+=1
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    print("error : ",e)
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
    
    def _process_id_split_file(self):
        id_split={
            "train":[],
            "val":[],
            "test":[]
        }

        # split data
        with open(self.data_split_file, 'rb') as f:
            data_split = pickle.load(f)
            
        for i in tqdm(range(len(self))):

            name=self.get_ori_data(i).protein_filename.split("/")[0]
            if name in data_split["train"]:
                id_split["train"].append(i)
            elif name in data_split["valid"]:
                id_split["test"].append(i)

        with open(self.id_split_file, 'wb') as f:
            pickle.dump(id_split, f)
            
        
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    PocketLigandPairDataset(args.path)
