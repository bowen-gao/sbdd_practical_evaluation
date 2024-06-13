import os
import glob
import rdkit 
from rdkit import Chem
from tqdm import tqdm
import random

def filter_valid_samples(samples):
    ok_list=[]
    for sample in samples:
        # check if the sdf file is valid
        try:
            mol = Chem.MolFromMolFile(sample,sanitize=False)

            # try sanitize
            Chem.SanitizeMol(mol)

            # try kekulize
            Chem.Kekulize(mol)
            ok_list.append(sample)
        except:
            # print("error")
            continue
    print("valid samples: %d"%len(ok_list))
    return ok_list

def format_pocket2mol_output():
    # exps=glob.glob('/data/pocket2mol_data/sample_output/*')
    suffix="/data/AR_data/PCBA_sample_output/"
    exps=[
        'PDBBind-DUD_E_FLAPP_0.9'
    ]
    exps = [suffix + exp for exp in exps]
    for exp in exps:
        cases=glob.glob(exp+'/*')
        cases=[x for x in cases if os.path.isdir(x)]
        for case in tqdm(cases):
            output_sdf=os.path.join(exp,os.path.basename(case)+'.sdf')
            samples=glob.glob(case+'/SDF/*.sdf')

            samples=filter_valid_samples(samples)

            #sort samples
            samples.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            
            # sample 20 ,if less than 20, sample all
            if len(samples)>20:
                samples_20=random.sample(samples,20)
            else:
                samples_20=samples
            
            print(f'{len(samples)} -> {len(samples_20)}')

            writer=Chem.SDWriter(output_sdf)
            for sample in samples_20:
                mol=Chem.MolFromMolFile(sample)
                if mol is not None:
                    writer.write(mol)
                else:
                    print('Error: %s'%sample)
            writer.close()

format_pocket2mol_output()