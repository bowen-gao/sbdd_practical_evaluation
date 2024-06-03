#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve


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



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    #logger.info("loading model(s) from {}".format(args.path))
    #state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    #model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    
    #names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)

    
    root_path = "/data/protein/DUD-E_ori/raw/all/"

    #root_path = "/data/protein/lib-pcba/raw/lit_pcba/"

    targets = os.listdir(root_path)
    auc_list = []
    bedroc_list = []
    ef_list = []
    
    max_sims = []

    fda_reps = task.encode_unimol_mols_once(model, "FDA_Approved.lmdb", "emb", "atoms", "coordinates")

    fda_reps = fda_reps[0]



    for target in tqdm(targets):
        
        mol_path = os.path.join(root_path, target, "mols.lmdb")
        emb_dir = os.path.join(root_path, target, "emb")
        # remove emb_dir
        #os.system(f"rm -rf {emb_dir}")
        
        mol_reps, mol_names, labels = task.encode_unimol_mols_once(model, mol_path, emb_dir, "atoms", "coordinates")

        # load pickle

        # with open(f"{root_path}{target}/drugclip_emb/mols.lmdb.pkl", "rb") as f:
        #     mol_reps, mol_names, labels = pickle.load(f)


        ligand_path = os.path.join(root_path, target, "ligand.lmdb")
        
        ligand_reps, ligand_names, _ = task.encode_unimol_mols_once(model, ligand_path, emb_dir, "atoms", "coordinates")

        test_path = "/drug/sbdd_bench/ar_pdbbind_0.9/mols"
        #test_path = "/drug/sbdd_bench/p2m_pcba/mols"
        
        generate_mols_path = os.path.join(test_path, target + ".lmdb")
        
        try:
            generate_reps, generate_names, _ = task.encode_unimol_mols_once(model, generate_mols_path, os.path.join(test_path, "emb"), "atoms", "coordinates")
        except:
            continue
        

        pocket_path = os.path.join(root_path, target, "pocket.lmdb")

        pocket_reps = task.encode_pocket_once(model, pocket_path)

    

        active_reps = mol_reps[labels == 1]
        
        #active_reps = mol_reps

        # calculate similarity
        #print(generate_reps.shape, pocket_reps.shape)
        sim = np.dot(ligand_reps, fda_reps.T)




        
 
    
        



def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
