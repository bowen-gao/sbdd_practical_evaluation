

from tqdm import tqdm

from rdkit import Chem

from rdkit.Chem import AllChem

from rdkit.Chem import DataStructs, rdMolDescriptors

import pickle

import numpy as np



from e3fp.pipeline import confs_from_smiles, fprints_from_mol
from e3fp.fingerprint.metrics import tanimoto

import logging

logging.getLogger().setLevel(logging.CRITICAL)

fprint_params = {'bits': 1024, 'radius_multiplier': 1.5, 'rdkit_invariants': True}






def similarity_generated_e3fp(target):
   
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
                
            fp = fprints_from_mol(mol, fprint_params=fprint_params)[0]
                
            fp_list.append(fp)


    actives_path = os.path.join(root_path, target, "fps", "actives_e3fp.pkl")
    
    
    actives_fp = pickle.load(open(actives_path, "rb"))


    fda_fp = pickle.load(open("fda_e3fp.pkl", "rb"))


    distance_list_actives = []
    for fp_gen in fp_list:
        distance_list_gen = []
        for fp_list in actives_fp:
            for fp in fp_list:
                sim = tanimoto(fp_gen, fp)
                distance_list_gen.append(sim)
        distance_list_actives.append(np.max(distance_list_gen))
    distance_list_actives = np.array(distance_list_actives)

    distance_list_fda = []
    for fp_gen in fp_list:
        distance_list_gen = []
        for fp_list in fda_fp:
            for fp in fp_list:
                sim = tanimoto(fp_gen, fp)
                distance_list_gen.append(sim)
        distance_list_fda.append(np.max(distance_list_gen))
    
    distance_list_fda = np.array(distance_list_fda)
    
    return np.mean(distance_list_actives), np.mean(distance_list_fda), target


def similarity_generated_morgan(target):
    
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
    suppl = Chem.SDMolSupplier(os.path.join(output_path, generated_file))

    fp_list = []
    for mol in suppl:
        if mol is not None:
                
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                
            fp_list.append(fp)


    actives_path = os.path.join(root_path, target, "fps", "actives_morgan_fp.pkl")
    
    
    actives_fp = pickle.load(open(actives_path, "rb"))


    
    fda_fp = pickle.load(open("fda_morgan_fp.pkl", "rb"))


    distance_actives_list = []
    for fp_gen in fp_list:
        distance_list_gen = []
        for fda in actives_fp:
            sim = DataStructs.FingerprintSimilarity(fp_gen, fda)
            distance_list_gen.append(sim)
        distance_actives_list.append(np.max(distance_list_gen))
    
    distance_fda_list = []

    for fp_gen in fp_list:
        distance_list_gen = []
        for fda in fda_fp:
            sim = DataStructs.FingerprintSimilarity(fp_gen, fda)
            distance_list_gen.append(sim)
        distance_fda_list.append(np.max(distance_list_gen))

    #distance_list = np.array(distance_list)
    
    return np.mean(distance_actives_list), np.mean(distance_fda_list), target


if __name__ == "__main__":
    
    
    import os
    root_path = "./DUD-E"
    
    output_path = "path/to/output_dir"

    targets = os.listdir(root_path)
    
    sims = []

    res_dic_actives= {}

    res_dic_fda = {}


    tbar = tqdm(total=len(targets))
    def call_back(results):
        if results is None:
            return
        res_acives, res_fda, target = results

        res_dic_actives[target] = res_acives
        res_dic_fda[target] = res_fda

    import multiprocessing as mp
    pool = mp.Pool(101)
    for jobs in targets:
        pool.apply_async(func=similarity_generated_morgan, args= (jobs.strip(),), callback=call_back)
    pool.close()
    pool.join()

    # print mean of all targets

    print("actives max similarity:", np.mean(list(res_dic_actives.values())))

    print("FDA max similarity:", np.mean(list(res_dic_fda.values())))
        
        

