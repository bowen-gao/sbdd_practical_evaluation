import os
import torch
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

class Retriever():
    def __init__(self, ebds_dir):
        self.ebds_dir=ebds_dir
        self.ebds,self.names = self._load_ebds()

    def _load_ebds(self):
        data=torch.load(self.ebds_dir)
        pocket_ebds=[]
        names=[]
        for item in tqdm(data):
            pocket_ebds.append(item['pocket_ebd'])
            names.append(item['pocket_name'])
        return torch.stack(pocket_ebds).to("cuda:1"),names

    def retrieve(self,query_ebd,top_k=None):
        '''
        Function to retrieve top k pockets given query embedding

        Args:
            query_ebd (torch.Tensor): Query embedding
            top_k (int, optional): Number of pockets to retrieve. Defaults to all.

        Returns:

            scores (torch.Tensor): Similarity score of the top k pockets
            idxs (torch.Tensor): Index of the top k pockets
            names (list): Names of the top k pockets
        '''
        if top_k is None:
            top_k=len(self.ebds)
        scores = torch.matmul(query_ebd,self.ebds.T)
        scores,idxs = torch.topk(scores,top_k)
        names = [self.names[idx] for idx in idxs]
        return scores,idxs,names
    
    def retrieve_all(self,query_reader,top_k=5):
        all_results=[]
        for query_ebd,query_name in tqdm(zip(query_reader.query_ebds,query_reader.query_names),total=len(query_reader.query_ebds)):
            scores,idxs,names = self.retrieve(query_ebd,top_k)
            results={'query_name':query_name,'top_k_pockets':names,'scores':scores}
            all_results.append(results)
        return all_results
    
    def _draw_score_dist_single(self,query_reader,name,output_dir):
        query_ebd = query_reader.get_ebd(name)
        scores,idxs,names = self.retrieve(query_ebd)
        output_dir=os.path.join(output_dir,name+".png")
        plt.clf()
        plt.hist(scores.cpu().numpy(),bins=100)
        plt.title(name)
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.savefig(output_dir)

    
    def draw_score_dist(self,query_reader,names,output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for name in names:
            self._draw_score_dist_single(query_reader,name,output_dir)


    
class QueryReader():
    def __init__(self,query_dir):
        self.query_dir=query_dir
        self.query_ebds,self.query_names = self._load_querys()
        self.name2ebd={name:ebd for name,ebd in zip(self.query_names,self.query_ebds)}

    def _load_querys(self):
        data=torch.load(self.query_dir)
        query_ebds=[]
        names=[]
        for item in data:
            query_ebds.append(item['pocket_ebd'])
            names.append(item['pocket_name'])
        return list(torch.stack(query_ebds).to("cuda:1")),names

    def __len__(self):
        return len(self.query_ebds)
    
    def __getitem__(self,idx):
        return self.query_ebds[idx],self.query_names[idx]

    def get_ebd(self,name):
        return self.name2ebd[name]
    

retriever=Retriever("/data/rag/ProFSA_ebds.pt")
query_reader=QueryReader("/data/rag/pdbbind_ebds.pt")
# results=retriever.retrieve_all(query_reader)
# torch.save(results,"/data/rag/retrieval_results.pt")
case_list=['1j14',
    '6msn',
    '6csr',
    '4qh7',
    '1ppk',
    '4fai',
    '3ce0',
    '5vdo',
    '4qjp',
    '6g97',
    '5qc4',
    '3v01']
retriever.draw_score_dist(query_reader,case_list,"/data/rag/ret_score_dist/")
