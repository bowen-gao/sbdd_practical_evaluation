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
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
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



    method_ef_list = []

    test_path_list = ["/drug/sbdd_bench/ligan_pdbbind_0.9", "/drug/sbdd_bench/ar_pdbbind_0.9", "/drug/sbdd_bench/p2m_pdbbind_0.9", "/drug/sbdd_bench/td_pdbbind_0.9", "/drug/sbdd_bench/bfn_pdbbind_0.9"]

    for path in test_path_list:
        ef_list = []
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

            #test_path = "/drug/sbdd_bench/ligan/mols"
            #test_path = "/drug/sbdd_bench/p2m_pcba/mols"

            test_path = path+"/mols"


            
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
            import numpy as np
            sim = np.dot(generate_reps, mol_reps.T)

            # sim = np.max(sim, axis=1)

            # sim = np.mean(sim)



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

        method_ef_list.append(ef_list) 
        
        
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Step 2: Collect data (example data, replace with your actual data)
    np.random.seed(42)
    method_list = ["ligan", "AR", "pocket2mol", "targetdiff", "bfn"]

    # Example data for enrichment factors for each method
    ef_data = {
        "LiGAN": method_ef_list[0],
        "AR": method_ef_list[1],
        "Pocket2Mol": method_ef_list[2],
        "TargetDiff": method_ef_list[3],
        "MolCRAFT": method_ef_list[4]
    }



    # Step 3: Create DataFrame
    ef_list = []
    for method in ef_data:
        ef_list.extend([(method, ef) for ef in ef_data[method]])

    df_ef = pd.DataFrame(ef_list, columns=["Method", "Enrichment Factor"])

    # Step 4: Remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
        return df.loc[filter]

    # Apply outlier removal for each method
    df_ef_filtered = df_ef.groupby("Method").apply(lambda x: remove_outliers(x, "Enrichment Factor")).reset_index(drop=True)


    # change the method to original order

    method_order = ["LiGAN", "AR", "Pocket2Mol", "TargetDiff", "MolCRAFT"]
    df_ef_filtered["Method"] = pd.Categorical(df_ef_filtered["Method"], categories=method_order, ordered=True)


    # min value of ef should be zero 

    print(df_ef_filtered["Enrichment Factor"].min())

    # Step 4: Generate Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Method", y="Enrichment Factor", data=df_ef_filtered)
    #plt.ylim(0, 20)

    # change the color of different methods

    colors = ["#FFA07A", "#20B2AA", "#87CEFA", "#778899", "#FFD700"]

    for i, patch in enumerate(plt.gca().artists):
        patch.set_facecolor(colors[i])

    #plt.title("Violin Plot of Enrichment Factors for Different Methods", fontsize=20)

    plt.ylabel("Enrichment Factor", fontsize=20)
    plt.grid(True)


    # set the font size of x and y ticks

    plt.xticks(fontsize=20)

    # rotate

    #plt.xticks(rotation=45)

    # remove x label

    plt.xlabel("")

    plt.savefig("violin_plot_ef_unimol.png", dpi = 300, bbox_inches = 'tight')
    

    
    
        



def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
