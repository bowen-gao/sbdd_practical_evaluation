# sbdd_practical_evaluation


## Dataset

The dataset is hosted at [HuggingFace Dataset Dir](https://huggingface.co/datasets/bgao95/Practical_SBDD)

It should contain following files:
### PDBBind.lmdb.zip

processed pdbbind data for training

### PDBBind-DUD_E_FLAPP_0.6.pkl

split file for 0.6 version

### PDBBind-DUD_E_FLAPP_0.9.pkl

split file for 0.9 version

### DUDE.zip

DUD-E test set

### LIT-PCBA.zip

LIT-PCBA test set

### DUDE_generated_mols.zip

generated molecules by different methods for targets in DUD-E

### LITPCBA_generated_mols.zip

generated molecules by different methods for targets in LIT-PCBA


### Model Training and Sampling



## Model Evaluation

All the generated Mols are in DUDE_generated_mols.zip and LITPCBA_generated_mols.zip. 

For DUD-E test:




for LIT-PCBA test






