import os
import torch
import glob
from tqdm import tqdm

class PocketEbdsFile():
    '''
    PocketEbdsFile is a class for loading a single pocket embeddings file output by encode.py of ProFSA
    '''
    def __init__(self,path) -> None:
        self.path = path
        raw_data=torch.load(self.path)
        self.data=[]
        for batch in tqdm(raw_data):
            batch_size = len(batch['pocket_emb'])
            for id in range(batch_size):
                piece = {
                    "pocket_ebd": batch['pocket_emb'][id],
                    "mol_ebd" : batch['mol_emb'][id],
                    "pocket_name": batch['pocket_name'][id],
                }
                self.data.append(piece)
        print("Loaded",self.path, "with",len(self.data),"pockets")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_data(self):
        return self.data

class PockerMerger():
    def __init__(self,ebds_dir,output_dir) -> None:
        self.ebds_dir=ebds_dir
        self.output_dir=output_dir
        print(os.path.join(self.ebds_dir,'*.pt'))
        output_files=glob.glob(os.path.join(self.ebds_dir,'*.pt'))
        output_files.sort()
        print(output_files)
        data_list=[PocketEbdsFile(file) for file in output_files]
        self.data=self._merge(data_list)
    
    def _merge(self,data_list) -> list:
        self.data=[]
        for data in data_list:
            self.data+=data.get_data()
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def save(self):
        torch.save(self.data,self.output_dir)
        print("Saved to",self.output_dir,"with",len(self.data),"pockets")
    
pocket_merger=PockerMerger("/data/rag/pocket_ebds/",'/data/rag/pdbbind_ebds.pt')
pocket_merger.save()
