# sbdd_practical_evaluation


## Dataset

The dataset is hosted at [HuggingFace Dataset Dir](https://huggingface.co/datasets/bgao95/Practical_SBDD)

It should contain following files:
### PDBBind.lmdb.zip

processed pdbbind data for training in lmdb format

### PDBBind-DUD_E_FLAPP_0.6.pkl

split file for 0.6 version

### PDBBind-DUD_E_FLAPP_0.9.pkl

split file for 0.9 version

### DUDE.zip

DUD-E test set

### LIT-PCBA.zip

LIT-PCBA test set

### DUDE_generated_mols.zip

generated molecules by different methods for targets in DUD-E. 

### PCBA_generated_mols.zip

generated molecules by different methods for targets in LIT-PCBA

## pretrain_weights.zip

drugclip.pt: weights for pretrained DrugCLIP model

mol_pre_no_h_220816.pt: weights for pretrained Uni-Mol molecular Encoder

pocket_pre_220816.pt: weights for pretrained Uni-Mol pocketr Encoder

### Model Training and Sampling



## Model Evaluation

All the generated Mols are in DUDE_generated_mols.zip and PCBA_generated_mols.zip. 


### Using fingerprints to do evaluation

```bash
cd Fingerprint_Eval
```

to do similaritiy based eval on DUD-E dataset

```bash
python sim_dude.py
```

to do similaritiy based eval on LIT-PCBA dataset

```bash
python sim_pcba.py
```

to do virtual screening eval on DUD-E dataset

```bash
python vs_dude.py
```





bash test_sbdd.sh




### Using Deep Encoders to do evaluation

```bash

cd Encoder_Eval

bash test_sbdd.sh

```

Note that the pretrained weights in Hugging Face dataset dir should be downloaded.

change the parametes in test_sbdd.sh

change encoder to drugclip or unimol

change metric to vs(virtual screening), sim(similarities), or score(DrugCLIP score)

change test path to point to DUD-E or LIT-PCBA path downloaded from Hugging Face Dir

change model path to point to the model outputs downloaded from Hugging Face Dir









