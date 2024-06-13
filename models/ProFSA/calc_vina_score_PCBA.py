
import concurrent.futures
from unittest import result
import torch
import rdkit
from rdkit import Chem
from tqdm import tqdm
import sys 
import os
import argparse
import glob
sys.path.append("/project/Pocket2Mol")
from evaluation.docking_vina import VinaDockingTask


def calc_vina_dock(mol, protein_root):
    vina_task = VinaDockingTask.from_generated_mol(
        mol, protein_root=protein_root,
        tmp_dir="/data/vina_tmp"
        )
    try:
        docking_results = vina_task.run(mode='dock', exhaustiveness=8)
    except Exception as e:
        return None
    return docking_results[0]['affinity']

class EvaluaionRunner:
    def __init__(self, args):
        self.sdf_path = args.sdf_path
        self.result_path=os.path.join(self.sdf_path, "vina_dock_results")
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.protein_root_prefix="/data/lit_pcba/raw"
        self.sdf_path = args.sdf_path
        self.num_process = args.num_process

    def evaluate_case(self, cases):
        for case in cases:
            print(f"Start evaluating {case}")
            case_name=os.path.basename(case).split(".")[0]

            if "_" in case_name:
                a,b=case_name.split("_")
                if a.isdigit():
                    case_name=b
                else:
                    case_name=a

            # pcba
            protein_root = glob.glob(os.path.join(self.protein_root_prefix,case_name,"*_pocket10A.pdb"))[0]
            print(protein_root)
            results=[]
            reader=Chem.SDMolSupplier(case)
            for i, mol in tqdm(enumerate(reader)):
                if mol is None:
                    print("Error: %s"%case)
                    continue
                    
                vina_dock_score=calc_vina_dock(mol, protein_root)
                if vina_dock_score is None:
                    print("Error: %s"%case_name)
                    continue
                results.append({
                    "mol": mol,
                    "vina_dock": vina_dock_score
                })

            result_path = os.path.join(self.result_path, case_name+".pt")
            torch.save(results, result_path)
            print(f"Finish evaluating {case}")

    def run(self):
        tasks = glob.glob(self.sdf_path + "/*.sdf")
        print(f"Total number of cases: {len(tasks)}")

        task_splits = [tasks[i::self.num_process] for i in range(self.num_process)]
        print("tasks_splits",task_splits)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_process) as executor:
            futures = []
            for i in range(self.num_process):
                future = executor.submit(self.evaluate_case, task_splits[i])
                futures.append(future)

            concurrent.futures.wait(futures)
        
        print("Finish all cases")
        
        # analyze results
        vina_dock_list=[]
        all_results=glob.glob(self.result_path+"/*.pt")
        for result_path in all_results:
            results=torch.load(result_path)
            vina_dock_list+=[x["vina_dock"] for x in results]

        print("#"*50)
        print("sdf_path: ", self.sdf_path)
        print(f"Vina dock score: {sum(vina_dock_list)/len(vina_dock_list)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf_path', type=str, default='/data/ligan_data/pretrained_PCBA')
    parser.add_argument('-n', '--num_process', type=int, default=15)
    args = parser.parse_args()

    runner = EvaluaionRunner(args)
    runner.run()
