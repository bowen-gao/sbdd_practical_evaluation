import os
import torch

class CaseSelector():
    '''
    This class is used to copy the pdbbind and profsa cases (pdb) file to a specific directory, for convenience of verifying the retrieval works.
    '''
    def __init__(self,ProFSA_dir,pdbbind_dir,ret_res_dir,output_dir):
        self.ProFSA_dir=ProFSA_dir
        self.pdbbind_dir=pdbbind_dir
        self.ret_res_dir=ret_res_dir
        self.output_dir=output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._load_results()

    def _load_results(self):
        data=torch.load(self.ret_res_dir)
        self.ret_result = {}
        for item in data:
            self.ret_result[item['query_name']] = item['top_k_pockets']
    
    def _copy_file(self,src,dst):
        os.system('cp '+src+' '+dst)
    
    def _copy_dir(self,src,dst):
        os.system('cp -r '+src+' '+dst)

    def _copy_pdbbind(self,pdb_name,output_dir):
        pdb_path=os.path.join(self.pdbbind_dir,pdb_name)
        self._copy_dir(pdb_path,output_dir)
        
    
    def _copy_profsa(self,pdb_name,output_dir):
        pdb_path=os.path.join(self.ProFSA_dir,pdb_name+'.pdb')
        self._copy_file(pdb_path,output_dir)

    def copy_case(self,case_name):
        output_dir=os.path.join(self.output_dir,case_name)
        if os.path.exists(output_dir):
            os.system('rm -r '+output_dir)
        self._copy_pdbbind(case_name,output_dir)
        ret_dir=os.path.join(output_dir,'retrieved_pdbs')
        os.makedirs(ret_dir)
        for item in self.ret_result[case_name]:
            self._copy_profsa(item,ret_dir)

    
case_selector = CaseSelector(
    ProFSA_dir='/drug/prot_frag/ligand_pocket_new/',
    pdbbind_dir='/data/pdbbind_2020/combine_set/',
    ret_res_dir='/data/rag/retrieval_results.pt',
    output_dir='/data/rag/retrieval_cases',
)
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

for case in case_list:
    case_selector.copy_case(case)