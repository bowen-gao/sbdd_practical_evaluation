import os
import sys
from tqdm import tqdm
from Bio.PDB import PDBParser, Selection, PDBIO
from Bio.PDB import MMCIFParser
from Bio.Data import IUPACData
from multiprocessing import Pool


aa_3_to_1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
     'MSE':'M','CME':'C','CSO':'C'}

class FastaExtractor:
    '''
    extract fasta file from dataset, save the fasta file to the dataset.fasta_file
    
    default is suitable for PDBBind dataset
    '''
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def run(self):
        # self._check_single_chain()
        self._get_fasta_file()
    
    def _check_single_chain(self):
        '''Check if the pockets contain multiple chains'''
        multiple_chain = 0
        for item in tqdm(self.dataset.get_items()):
            pdbparser = PDBParser()
            structure = pdbparser.get_structure('item', item['pocket_dir'])
            model = structure[0]
            chain_num=0
            for chain in model:
                if chain.get_id() == " ":
                    continue
                print("chain id: ", chain.get_id())
                chain_num+=1
            if chain_num!=1:
                multiple_chain += 1
            print(chain_num)
        print("Multiple chain pockets: ", multiple_chain)

    def _get_fasta_file(self):
        '''get fasta file of all the related protein chains in dataset'''

        if os.path.exists(self.dataset.fasta_file):
            print("Fasta file exists: ", self.dataset.fasta_file)
            return
        
        seq_list=[]
        for item in tqdm(self.dataset.get_items()):
            pdbparser = PDBParser(QUIET=True)
            structure = pdbparser.get_structure('item', item['pocket_dir'])
            model = structure[0]
            for chain in model:
                if chain.get_id() == " ":
                    continue
                chain_name,chain_seq=self._get_chain_seq_by_id(item["protein_dir"],chain.get_id())
                if chain_name==None or chain_seq==None:
                    continue
                seq_list.append((chain_name,chain_seq))
        
        self._save_fasta(seq_list, self.dataset.fasta_file)

    def _get_chain_seq_by_id(self, protein, chain_id="all"):
        '''get a chain sequence by chain id and protein dir'''
        print("processing: ", protein, chain_id)
        script_path="/data/rosetta.binary.linux.release-315/main/tools/protein_tools/scripts/clean_pdb.py"
        if chain_id == "all":
            chain_id = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        cmd="python "+script_path+" "+protein+" "+chain_id
        res=os.popen(cmd).read().strip().split("\n")
        if len(res)!=5:
            print("!"*50)
            print("Error in get_chain_seq_by_id")
            return None,None
        chain_name=res[2].strip()[1:]
        chain_seq=res[3].strip()
        return chain_name,chain_seq
        
    
    def _save_fasta(self, seq_list, file_dir=None):
        '''save the fasta file to the dir
        Args:
            seq_list: list of tuple, (chain_name, chain_seq)
            file_dir: the dir to save the fasta file
        '''
        if file_dir is None:
            file_dir = self.dataset.get_tmp_file_dir()+"/fasta.txt"
        with open(file_dir, 'w') as f:
            for chain_name, chain_seq in seq_list:
                f.write(">"+chain_name+"\n")
                f.write(chain_seq+"\n")
        print("Fasta file saved in: ", file_dir)


class DUDEFastaExtractor(FastaExtractor):
    '''
    extract fasta file from DUDE dataset, save the fasta file to the dataset.fasta_file
    '''
    
    def __init__(self, dataset):
        super().__init__(dataset)
        self.target_info_dir="/data/DUD-E/target_info.txt"

        with open(self.target_info_dir) as f:
            lines=f.readlines()
        self.dude_pdb_map={}
        for line in lines[1:]:
            if line.strip()=="":
                continue
            line=line.strip().split("\t")
            self.dude_pdb_map[line[0]]=line[7]

    def _get_chain_seq_by_id(self, protein_dir, chain_id="all"):
        '''get a chain sequence by chain id and protein dir'''
        
        script_path="/drug/rosetta.binary.linux.release-315/main/tools/protein_tools/scripts/clean_pdb.py"
        if chain_id == "all":
            chain_id = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        cmd="python "+script_path+" "+protein_dir+" "+chain_id
        res=os.popen(cmd).read().strip().split("\n")
        chain_names=[]
        chain_seqs=[]
        for i in range(3,len(res),2):
            chain_names.append(res[i].strip()[1:])
            chain_seqs.append(res[i+1].strip())
        return chain_names,chain_seqs
        

    def _get_fasta_file(self):
        seq_list=[]
        for item in tqdm(self.dataset.get_items()):
            dude_name=item['name']
            pdb_id=self.dude_pdb_map[dude_name]
            chain_names,chain_seqs=self._get_chain_seq_by_id(pdb_id)
            for chain_name,chain_seq in zip(chain_names,chain_seqs):
                chain_name=chain_name.replace(pdb_id.upper(),dude_name)
                seq_list.append((chain_name,chain_seq))
        self._save_fasta(seq_list, self.dataset.fasta_file)
            
            


class BioLipFastaExtractor(FastaExtractor):
    '''
    extract fasta file from BioLip dataset, save the fasta file to the dataset.fasta_file
    '''
    
    def __init__(self, dataset):
        super().__init__(dataset)


    def _get_fasta_file(self):
        seq_list=[]
        tasks=[]
        results=[]
        for item in tqdm(self.dataset):
            protein_id,_,chain_id,_=item['name'].split("_")
            tasks.append((item['protein_dir'],chain_id))

        # for task in tasks:
        #     chain_seq=self._get_chain_seq_by_id(*task)
        #     results.append(chain_seq)
        with Pool(10) as p:
            results = p.starmap(self._get_chain_seq_by_id, tasks)

        for i in range(len(tasks)):
            seq_list.append((self.dataset.get_items()[i]['name'],results[i]))
        self._save_fasta(seq_list, self.dataset.fasta_file)


    def _get_chain_seq_by_id(self, protein, chain_id="all"):
        ''' read .cif using Bio and extractor the chain 1-letter expression'''
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('item', protein)
        model = structure[0]
        chain = model[chain_id]
        seq=""
        for residue in chain:
            if residue.get_id()[0] == " ":
                residue_name=residue.get_resname()
                if residue_name in aa_3_to_1:
                    seq += aa_3_to_1[residue_name]
                else : 
                    print("Find non-standard residue: ",residue_name)
                
        return seq