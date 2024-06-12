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
    
    task = tasks.setup_task(args)
    model = task.build_model(args)

    if args.encoder == "drugclip":
        state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
        model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    

    root_path = args.testset 


    targets = os.listdir(root_path)
    auc_list = []
    bedroc_list = []
    ef_list = []
    
    max_actives_sims = []
    max_fda_sims = []

    scores = []

    if args.encoder == "unimol":
        fda_reps = task.encode_unimol_mols_once(model, "FDA_Approved.lmdb", "emb", "atoms", "coordinates")
    else:
        fda_reps = task.encode_mols_once(model, "FDA_Approved.lmdb", "drugclip_emb", "atoms", "coordinates")

    fda_reps = fda_reps[0]



    for target in tqdm(targets):
        
        mol_path = os.path.join(root_path, target, "mols.lmdb")
        if args.encoder == "unimol":
            emb_dir = os.path.join(root_path, target, "emb")
        else:
            emb_dir = os.path.join(root_path, target, "drugclip_emb")

        if args.encoder == "unimol":
            mol_reps, mol_names, labels = task.encode_unimol_mols_once(model, mol_path, emb_dir, "atoms", "coordinates")
        else:
            mol_reps, mol_names, labels = task.encode_mols_once(model, mol_path, emb_dir, "atoms", "coordinates")




        test_path = args.model + "/mols"
        
        generate_mols_path = os.path.join(test_path, target + ".lmdb")
        
        try:
            if args.encoder == "unimol":
                generate_reps, generate_names, _ = task.encode_unimol_mols_once(model, generate_mols_path, os.path.join(test_path, "emb"), "atoms", "coordinates")
            else:
                generate_reps, generate_names, _ = task.encode_mols_once(model, generate_mols_path, os.path.join(test_path, "drugclip_emb"), "atoms", "coordinates")
        except:
            continue
        
        if os.path.exists(os.path.join(root_path, target, "pocket.lmdb")):
            pocket_path = os.path.join(root_path, target, "pocket.lmdb")
        else:
            pocket_path = os.path.join(root_path, target, "pockets.lmdb")



        pocket_reps = task.encode_pocket_once(model, pocket_path)

    

        active_reps = mol_reps[labels == 1]
        
        if args.metric == "sim":
            sim_actives = np.dot(generate_reps, active_reps.T)
            sim_actives = np.max(sim_actives, axis=1)
            sim_actives = np.mean(sim_actives)
            max_actives_sims.append(sim_actives)
            sim_fda = np.dot(generate_reps, fda_reps.T)
            sim_fda = np.max(sim_fda, axis=1)
            sim_fda = np.mean(sim_fda)
            max_fda_sims.append(sim_fda)
        
        elif args.metric == "score":
            score = np.dot(generate_reps, pocket_reps.T)
            score = np.max(score, axis=1)
            score = np.mean(score)
            scores.append(score)
        
        elif args.metric == "vs":
            sim = np.dot(generate_reps, mol_reps.T)
            aucs = []
            efs  = []
            bedrocs = []
            for i in range(sim.shape[0]):
                auc, bedroc, ef = cal_metrics(labels, sim[i])
                aucs.append(auc)
                bedrocs.append(bedroc)
                efs.append(ef)
            
            auc_list.append(np.mean(aucs))
            bedroc_list.append(np.mean(bedrocs))
            ef_list.append(np.mean(efs))

                




        



        
        

    if args.metric == "sim":
        print("Max actives sims: ", np.mean(max_actives_sims))
        print("Max FDA sims: ", np.mean(max_fda_sims))
    
    elif args.metric == "score":
        print("DrugCLIP Score: ", np.mean(scores))
    
    elif args.metric == "vs":
        print("BEDROC: ", np.mean(bedroc_list))
        print("EF: ", np.mean(ef_list))



    
        



def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    options.add_model_args(parser)


    # add args
    parser.add_argument("--encoder", type=str, default="unimol", help="encoder model")
    parser.add_argument("--metric", type=str, default="sim", help="mode")
    parser.add_argument("--testset", type=str, default="", help="testset")
    parser.add_argument("--model", type=str, default="", help="model path")

    args = options.parse_args_and_arch(parser)


    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
