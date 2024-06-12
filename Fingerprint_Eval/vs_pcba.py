
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
            




def similarity_generated(root_path, output_path, target):
    


    files = os.listdir(output_path)
    

    if files[0][0].isdigit():
        target_2_file = {}
        files = os.listdir(output_path)
        for file in files:
            if file.endswith("sdf"):
                tar = file.split("_")[1].split(".")[0]
                target_2_file[tar] = file
        
        try:

            generated_file = target_2_file[target]
        except:
            return None
    else:
        generated_file = target + ".sdf"

    # read sdf
    suppl = Chem.SDMolSupplier(os.path.join(output_path, generated_file))

    fp_list = []
    for mol in suppl:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp_list.append(fp)



    


   
    
    active_fp_path = os.path.join(root_path, target, "fps", "actives_morgan_fp.pkl")

    active_fps = pickle.load(open(active_fp_path, "rb"))
    
    
    res_actives = []
    for fp in tqdm(fp_list):

        # calculate Tanimoto similarity

        sims = []
        for i in range(len(active_fps)):
            sim = DataStructs.FingerprintSimilarity(active_fps[i] , fp)
            sims.append(sim)
        res_actives.append(np.max(sims))
    
    res_fda = []

    fda_fp = pickle.load(open("fda_morgan_fp.pkl", "rb"))

    for fp in tqdm(fp_list):
            
        # calculate Tanimoto similarity

        sims = []
        for i in range(len(fda_fp)):
            sim = DataStructs.FingerprintSimilarity(fda_fp[i] , fp)
            sims.append(sim)
        res_fda.append(np.max(sims))
            
    
    return np.mean(res_actives), np.mean(res_fda)




        

if __name__ == "__main__":


    root_path = "./LIT-PCBA"

    output_path = "path/to/output_dir"
    targets = os.listdir(root_path)
    auc_list = []
    bedroc_list = []
    ef_list = []

    sims_actives = []
    sims_fda = []
    for i,target in enumerate(tqdm((targets))):

        sims = similarity_generated(root_path, output_path, target)
        if sims is not None:
            sim_actives, sim_fda = sims
            sims_actives.append(sim_actives)
            sims_fda.append(sim_fda)

    print("Actives max similarity:", np.mean(sims_actives))
    print("FDA max similarity:", np.mean(sims_fda))



    