import os
import sys
sys.path.append(".")
from scripts.benchmark.dataset import PDBBindDataset,DUDEDataset
import pickle

class targetdiff_index_generator:
    '''
    this generator is used to generate index.pkl for different datasplits to train with targetdiff
    '''
    def __init__(self):
        self.index=[]
    def run(self,dataset,output_path):
        self.output_path=output_path

        self.dataset=dataset
        for item in self.dataset.get_items():
            pocket_fn=item['name']+"/"+item['pocket_dir'].split("/")[-1]
            ligand_fn=item['name']+"/"+item['ligand_dir'].split("/")[-1]
            self.index.append((pocket_fn,ligand_fn))
        with open(self.output_path,"wb") as f:
            pickle.dump(self.index,f)
        
if __name__=="__main__":
    output_path="/data/DUD-E/targetdiff_index_files/DUD-E6A.pkl"
    generator=targetdiff_index_generator()
    dataset=DUDEDataset()
    generator.run(dataset,output_path)