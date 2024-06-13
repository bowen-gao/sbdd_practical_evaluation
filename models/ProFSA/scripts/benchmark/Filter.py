from concurrent.futures import thread
from importlib.machinery import FrozenImporter
import os
import abc
from Bio.PDB import PDBParser
from tqdm import tqdm
from fasta_extractor import BioLipFastaExtractor, FastaExtractor, DUDEFastaExtractor
from dataset import BioLipDataset, PDBBindDataset, DUDEDataset
from rdkit.Chem import DataStructs,AllChem
from rdkit import Chem
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices as sm
from multiprocessing import Pool



class FilterGenerator:

    @abc.abstractmethod
    def generate(self):
        pass

    def _filter_dataset(self,thresholds=None,sizes=None):
        assert thresholds is not None or sizes is not None, "thresholds and sizes cannot be both None"
        assert thresholds is None or sizes is None, "thresholds and sizes cannot be both not None"

        # make sure score is float
        for i in range(len(self.scores_list)):
            self.scores_list[i]=(self.scores_list[i][0],self.scores_list[i][1],float(self.scores_list[i][2]))
        # sort 
        self.scores_list.sort(key=lambda x: x[2],reverse=True)

        if sizes is not None:
            thresholds =None
        
        if thresholds is not None:
            for threshold in thresholds:
                remove_list=set()
                for item in self.scores_list:
                    if item[2]>=threshold:
                        remove_list.add(item[1])
                
                data=set(self.dataset_t.get_name_list())-remove_list
                output_name=self.dataset_t.dataset_name+"-"+self.dataset_s.dataset_name+"_"+self.type+"_"+str(threshold)+".pkl"
                output_name=os.path.join(self.output_dir,output_name)

                self._save_and_split_dataset(data,output_name)
                print(f"Split data: {self.dataset_t.dataset_name}-{self.dataset_s.dataset_name} {self.type} with threshold {threshold} : {output_name} ,size of data:{len(data)}")
        
        if sizes is not None:
            for size in sizes:
                remove_list=set()
                for item in self.scores_list:
                    if item[1] not in self.dataset_t.get_name_list():
                        continue
                    remove_list.add(item[1])
                    data_num=len(self.dataset_t.get_name_list())-len(remove_list)
                    if data_num==size:    
                        threashold=item[2]
                        break
                data=set(self.dataset_t.get_name_list())-remove_list
                output_name=self.dataset_t.dataset_name+"-"+self.dataset_s.dataset_name+"_"+self.type+"_"+str(size)+".pkl"
                output_name=os.path.join(self.output_dir,output_name)

                self._save_and_split_dataset(data,output_name)
                print(f"Split data: {self.dataset_t.dataset_name}-{self.dataset_s.dataset_name} {self.type} with size {size} : {output_name} ,size of data:{len(data)},threshold:{threashold}")


    def _save_and_split_dataset(self,data,output_name):
        '''
        split the data into train and valid with 9:1

        then save the train and valid data to the output_name with a pkl format: {train:[],valid:[]}
        '''
        import random
        data=list(data)
        random.shuffle(data)
        split_point=int(len(data)*0.9)
        train_data=data[:split_point]
        valid_data=data[split_point:]
        with open(output_name,'wb') as f:
            import pickle
            pickle.dump({"train":train_data,"valid":valid_data},f)



class SequenceSimilarityFilter(FilterGenerator):
    def __init__(self, dataset_s,dataset_t, similarity_output_file=None,output_dir="/data/rag/data_splits"):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.similarity_output_file = similarity_output_file
        self.output_dir=output_dir
        self.type="SeqSimilarity"

    def _generate_fasta(self,dataset):
        if isinstance(dataset,PDBBindDataset):
            exactor=FastaExtractor(dataset)
        elif isinstance(dataset,DUDEDataset):
            exactor=DUDEFastaExtractor(dataset)
        elif isinstance(dataset,BioLipDataset):
            exactor=BioLipFastaExtractor(dataset)
        else : 
            raise NotImplementedError
        exactor.run()

    def _read_fasta(self,fasta_file):
        ret=[]
        zeros=0
        with open(fasta_file,'r') as f:
            lines = f.readlines()
            for idx in range(0,len(lines),2):
                ret.append((lines[idx].strip()[1:],lines[idx+1].strip()))
                if len(lines[idx+1].strip())<=30:
                    print(lines[idx].strip(),len(lines[idx+1].strip()))
                    zeros+=1
        print("zeros:",zeros)
        return ret
    

    def _sequence_similarity(self,item_s,item_t):
        name_s,seq_s=item_s
        name_t,seq_t=item_t
        if len(seq_s)==0:
            print(f"Error in {name_s}")
        if len(seq_t)==0:
            print(f"Error in {name_t}")

        matrix = sm.load("BLOSUM62")
        aligner = PairwiseAligner()
        aligner.substitution_matrix = matrix
        aligner.open_gap_score = -2.0
        aligner.extend_gap_score = -2.0
        aligner.mode = "local"

        alignment = aligner.align(seq_s, seq_t)[0]
        score=alignment.score
        assert len(alignment)==2
        alignment_len=len(alignment[0])

        cnt=0
        _1,_2=0,0
        for i in range(alignment_len):
            aa_i,aa_j=alignment[0][i],alignment[1][i]
            if aa_i=="-":
                _1+=1
            if aa_j=="-":
                _2+=1
            if aa_i=="-" or aa_j=="-":
                score=0
            else:
                score=matrix[aa_i,aa_j]
            if score>0:
                cnt+=1
        alignment_len=max(_1,_2)+max(len(seq_s),len(seq_t))
        similarity=cnt/alignment_len

        # print(f"{name_s} and {name_t} similarity: {similarity}")
        # print(seq_s)
        # print(seq_t)
        return (name_s,name_t,similarity)

    def _run_sequence_similarity(self):
        
        # read sequences
        seqs_s=self._read_fasta(self.dataset_s.fasta_file)
        seqs_t=self._read_fasta(self.dataset_t.fasta_file)

        # assign tasks
        tasks=[]
        for item_s in seqs_s:
            for item_t in seqs_t:
                tasks.append((item_s,item_t))
        print(f"Total {len(tasks)} tasks")

        # results=[]
        # # run tasks
        # for task in tqdm(tasks):
        #     result=self._sequence_similarity(*task)
        #     results.append(result)
        
        # Create a pool of workers
        with Pool(256) as p:
            results = p.starmap(self._sequence_similarity, tasks)


        # save results
        with open(self.similarity_output_file,'w') as f:
            for result in results:
                f.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")

    
    def _process_raw(self):
        score_map={}
        with open(self.similarity_output_file_raw,'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line=line.strip().split("\t")
                source=line[0].split("_")[0]
                target=line[1]
                score=float(line[2])
                if (source,target) not in score_map:
                    score_map[(source,target)]=score
                else:
                    score_map[(source,target)]=max(score,score_map[(source,target)])
        
        with open(self.similarity_output_file,'w') as f:
            for k,v in score_map.items():
                f.write(f"{k[0]}\t{k[1]}\t{v}\n")
        
        print("Finish processing raw file, saved",len(score_map),"pairs of sequences")

    def run(self,thresholds=None,sizes=None):
        if not os.path.exists(self.dataset_s.fasta_file):
            self._generate_fasta(self.dataset_s)
        if not os.path.exists(self.dataset_t.fasta_file):
            self._generate_fasta(self.dataset_t)

        self.similarity_output_file_raw=self.similarity_output_file.split(".")[0]+"_raw.txt"
        if not os.path.exists(self.similarity_output_file_raw):
            print("Run sequence similarity")
            self._run_sequence_similarity()
        else :
            print("Sequence similarity raw file exists")
        
        if not os.path.exists(self.similarity_output_file):
            self._process_raw()
        
        self._parse_SeqSimilarity_output()
        self._filter_dataset(thresholds=thresholds,sizes=sizes)
        

    
    def _parse_SeqSimilarity_output(self):
        '''
        calc the self.scores_list

        self.scores_list: list of tuple (source,target,matching_score)
        '''

        with open(self.similarity_output_file, 'r') as f:
            lines = f.readlines()

        self.scores_list=[]
        for line in tqdm(lines):
            result_line=line.strip().split("\t")
            score=float(result_line[2])
            source = result_line[0].strip()
            target = result_line[1].strip()
            self.scores_list.append((source,target,score))






class FLAPPFilter(FilterGenerator):
    def __init__(self, dataset_s,dataset_t, FLAPP_output_file=None,output_dir="/data/rag/data_splits"):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.FLAPP_output_file = FLAPP_output_file
        self.output_dir=output_dir
        self.type="FLAPP"

    def _run_FLAPP(self):
        raise NotImplementedError

    def run(self,thresholds=None,sizes=None):
        if self.FLAPP_output_file is None:
            self._run_FLAPP()
        
        self._parse_FLAPP_output()
        self._filter_dataset(thresholds=thresholds,sizes=sizes)

    
    def _parse_FLAPP_output(self):
        '''
        calc the self.scores_list

        self.scores_list: list of tuple (source,target,matching_score)

        '''

        with open(self.FLAPP_output_file, 'r') as f:
            lines = f.readlines()

        self.scores_list=[]
        for line in tqdm(lines[1:]):
            result_line=line.strip().split("\t")
            scores=result_line[2].split(" ")
            matching_score=max(float(scores[3]),float(scores[4]))
            source = result_line[0].strip().split(".")[0]
            target = result_line[1].strip().split(".")[0]
            self.scores_list.append((source,target,matching_score))


class MorganFilter(FilterGenerator):
    def __init__(self, dataset_s,dataset_t, morgan_output_file=None,output_dir="/data/rag/data_splits"):
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t
        self.morgan_output_file = morgan_output_file
        self.output_dir=output_dir
        self.type="Morgan"

    def _morgan_similarity(self,name_s,name_t,mol_s_dir,mol_t_dir):
        try:
            if mol_s_dir.endswith(".mol2"):
                mol_s = Chem.MolFromMol2File(mol_s_dir)
            elif mol_s_dir.endswith(".sdf"):
                mol_s = Chem.MolFromMol2File(mol_s_dir)
            elif mol_s_dir.endswith(".pdb"):
                mol_s = Chem.MolFromPDBFile(mol_s_dir)
            else:
                raise NotImplementedError
            if mol_t_dir.endswith(".mol2"):
                mol_t = Chem.MolFromMol2File(mol_t_dir)
            elif mol_t_dir.endswith(".sdf"):
                mol_t = Chem.MolFromMol2File(mol_t_dir)
            elif mol_t_dir.endswith(".pdb"):
                mol_t = Chem.MolFromPDBFile(mol_t_dir)
            else:
                raise NotImplementedError
            
            fp_s=AllChem.GetMorganFingerprintAsBitVect(mol_s, 2, nBits=1024)
            fp_t=AllChem.GetMorganFingerprintAsBitVect(mol_t, 2, nBits=1024)
            sim=DataStructs.FingerprintSimilarity(fp_s,fp_t)
            return sim
        except Exception as e:
            print(f"Error in {mol_s_dir} and {mol_t_dir}:",e)
            return -1

    def _run_morgan(self):
        '''
        calc morgan fingerprint within pairs , save in self.morgan_output_file
        '''
        tasks=[]
        for item_s in self.dataset_s.get_items():
            for item_t in self.dataset_t.get_items():
                name_s=item_s['name']
                name_t=item_t['name']
                mol_s=item_s['ligand_dir']
                mol_t=item_t['ligand_dir']
                tasks.append((name_s,name_t,mol_s,mol_t))
        print(f"Total {len(tasks)} tasks")

        import multiprocessing
        pool = multiprocessing.Pool(256)
        results = pool.starmap(self._morgan_similarity, tasks)
        pool.close()
        pool.join()

        # results=[]
        # for task in tqdm(tasks):
        #     results.append(self._morgan_similarity(*task))

        with open(self.morgan_output_file,'w') as f:
            for i in range(len(tasks)):
                f.write(f"{tasks[i][0]}\t{tasks[i][1]}\t{results[i]}\n")


    def run(self,thresholds=None,sizes=None):
        if not os.path.exists(self.morgan_output_file):
            self._run_morgan()
        
        self._parse_morgan_output()
        self._filter_dataset(thresholds=thresholds,sizes=sizes)

    
    def _parse_morgan_output(self):
        '''
        calc the self.scores_list

        self.scores_list: list of tuple (source,target,matching_score)
        '''

        with open(self.morgan_output_file, 'r') as f:
            lines = f.readlines()

        self.scores_list=[]
        for line in tqdm(lines):
            result_line=line.strip().split("\t")
            score=result_line[2]
            source = result_line[0].strip()
            target = result_line[1].strip()
            self.scores_list.append((source,target,score))


