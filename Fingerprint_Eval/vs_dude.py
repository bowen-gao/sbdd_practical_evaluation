
import os
import rdkit

from rdkit import Chem

from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs, rdMolDescriptors
from tqdm import tqdm
import pickle

import numpy as np
from scipy import stats
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment

from sklearn.metrics import roc_auc_score

from e3fp.pipeline import confs_from_smiles, fprints_from_mol

from e3fp.fingerprint.metrics import tanimoto

import logging

logging.getLogger().setLevel(logging.ERROR)




def cal_metrics(y_true, y_score):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index  = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.01])
    ef = ef_list[0]
    return auc, bedroc, ef



            

def vs_generated_morgan(target):
    
    files = os.listdir(output_path)
    
    if files[0][0].isdigit():
        target_2_file = {}
        files = os.listdir(output_path)
        for file in files:
            if file.endswith("sdf"):
                tar = file.split("_")[1].split(".")[0]
                target_2_file[tar] = file


        #print(target_2_file)
        
        try:

            generated_file = target_2_file[target]
        except:
            return None
    else:
        generated_file = target + ".sdf"

    # read sdf
    try:
        suppl = Chem.SDMolSupplier(os.path.join(output_path, generated_file))
    except:
        return None
    fp_list = []
    mol_weight_list = []
    for mol in suppl:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp_list.append(fp)

            mol_weight_list.append(mol.GetNumAtoms())




   

    active_fp_path = os.path.join(root_path, target, "fps", "actives_morgan_fp.pkl")
    active_fps = pickle.load(open(active_fp_path, "rb"))
    
    decoy_fp_path = os.path.join(root_path, target, "fps", "decoys_morgan_fp.pkl")
    decoy_fps = pickle.load(open(decoy_fp_path, "rb"))
    
    auc_list = []
    bedroc_list = []
    ef_list = []


    for i in range(len(fp_list)):

        # calculate Tanimoto similarity
        fp = fp_list[i]


        sims = []

        labels = []

        for i in range(len(active_fps)):
            sim = DataStructs.FingerprintSimilarity(active_fps[i] , fp)
            sims.append(sim)
            labels.append(1)
        
        for i in range(len(decoy_fps)):
            sim = DataStructs.FingerprintSimilarity(decoy_fps[i] , fp)
            sims.append(sim)
            labels.append(0)

        
        auc, bedroc, ef = cal_metrics(labels, sims)
        auc_list.append(auc)
        bedroc_list.append(bedroc)
        ef_list.append(ef)
    
    return np.mean(auc_list), np.mean(bedroc_list), np.mean(ef_list), target


def vs_generated_e3fp(target):
    
    fprint_params={'bits': 1024, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
    
    files = os.listdir(output_path)
    
    # check whether files[0] start with number

    if files[0][0].isdigit():
        target_2_file = {}
        files = os.listdir(output_path)
        for file in files:
            if file.endswith("sdf"):
                tar = file.split("_")[1].split(".")[0]
                target_2_file[tar] = file


        #print(target_2_file)
        
        try:

            generated_file = target_2_file[target]
        except:
            return None
    else:
        generated_file = target + ".sdf"

    # read sdf
    try:
        suppl = Chem.SDMolSupplier(os.path.join(output_path, generated_file))
    except:
        return None
    fp_list = []
    for mol in suppl:
        if mol is not None:
            try:
                fp = fprints_from_mol(mol, fprint_params=fprint_params)[0]
            except:
                continue
            fp_list.append(fp)




   

    active_fp_path = os.path.join(root_path, target, "fps", "actives_e3fp.pkl")
    active_fps = pickle.load(open(active_fp_path, "rb"))
    
    decoy_fp_path = os.path.join(root_path, target, "fps", "decoys_e3fp.pkl")
    decoy_fps = pickle.load(open(decoy_fp_path, "rb"))
    
    auc_list = []
    bedroc_list = []
    ef_list = []
    

    for i in range(len(fp_list)):

        # calculate Tanimoto similarity
        fp = fp_list[i]


        sims = []

        labels = []

        for i in range(len(active_fps)):
            max_sim = 0
            for active_fp in active_fps[i]:
                sim = tanimoto(active_fp , fp)
                max_sim = max(max_sim, sim)

            #print(sim)
            sims.append(max_sim)
            labels.append(1)
        
        for i in range(len(decoy_fps)):
            max_sim = 0
            for decoy_fp in decoy_fps[i]:
                sim = tanimoto(decoy_fp , fp)
                max_sim = max(max_sim, sim)
            sims.append(max_sim)
            labels.append(0)
        
        auc, bedroc, ef = cal_metrics(labels, sims)
        auc_list.append(auc)
        bedroc_list.append(bedroc)
        ef_list.append(ef)
    
    return np.mean(auc_list), np.mean(bedroc_list), np.mean(ef_list), target




if __name__ == "__main__":



    root_path = "./DUD-E"
    
    output_path = "path/to/output_dir"
    
    targets = os.listdir(root_path)
    auc_list = []
    bedroc_list = []
    ef_list = []



    tbar = tqdm(total=len(targets))
    auc_dic = {}
    bedroc_dic = {}
    ef_dic = {}

    def call_back(results):
        if results is None:
            return
        auc, bedroc, ef, target = results

        auc_dic[target] = auc
        bedroc_dic[target] = bedroc
        ef_dic[target] = ef


        

    import multiprocessing as mp
    pool = mp.Pool(101)
    for jobs in targets:
        pool.apply_async(func=vs_generated_e3fp, args= (jobs.strip(),), callback=call_back)
    pool.close()
    pool.join()

    # print mean of all targets

    #print("AUC:" ,np.mean(list(auc_dic.values())))
    print("BEDROC:", np.mean(list(bedroc_dic.values())))
    print("EF:", np.mean(list(ef_dic.values())))

    




    
                



    



    





