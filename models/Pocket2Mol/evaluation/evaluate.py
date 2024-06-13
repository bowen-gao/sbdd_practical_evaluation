import os
import argparse
from copy import deepcopy
import torch
from tqdm.auto import tqdm
from rdkit.Chem.QED import qed
import sys
sys.path.append('.')

from utils.reconstruct import reconstruct_from_generated_with_edges
from sascorer import compute_sa_score
from evaluation.docking import *
from utils.misc import *
from evaluation.scoring_func import *
from evaluation.docking_vina import VinaDockingTask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--result_root', type=str, default='./outputs')
    parser.add_argument('--protein_root', type=str, default='./evaluation/pdb_data')
    parser.add_argument('--exhaustiveness', type=int, default=8)
    args = parser.parse_args()

    exp_dir = os.path.join(args.result_root, args.exp_name)
    save_path = os.path.join(exp_dir, 'samples_all.pt') # get_saved_filename(exp_dir))

    samples = torch.load(save_path, map_location='cpu')

    sim_with_train = SimilarityWithTrain()
    results = []
    for i, data in enumerate(tqdm(samples.finished, desc='All')):
        try:
            mol = reconstruct_from_generated_with_edges(data)

            vina_task = VinaDockingTask.from_generated_mol(
                    mol, protein_root=args.protein_root,
                    tmp_dir="/data/vina_tmp"
                    )
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
 
            results.append({
                'mol': mol,
                'vina_score': score_only_results,
                'vina_min': minimize_results,
                'qed': qed(mol),
                'sa': compute_sa_score(mol),
                'lipinski': obey_lipinski(mol),
                'logp': get_logp(mol),
            })
        except Exception as e:
            print('Failed %d' % i)


    result_path = os.path.join(exp_dir, 'results.pt')
    torch.save(results, result_path)
