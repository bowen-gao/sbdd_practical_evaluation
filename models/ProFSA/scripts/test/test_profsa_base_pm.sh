# export CUDA_VISIBLE_DEVICES="4,5,6,7"

# base
logdir=/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-08_21-48-13
# remove used parameters
logdir=/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-17_18-59-09
# logit scale = 32
logdir=/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-20_02-44-18
# seed = 1
logdir=/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-20_13-30-11
# profsa ori
logdir='/log/train/profsa/profsa_ckpt_modified.pt'
# new split (19)
logdir=/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-02-29_13-48-07

python test.py $logdir \
    --update_func test_kahraman test_toughm1 \
    --update_wandb

python test.py $logdir \
    --update_func test_kahraman test_toughm1 \
    --ckpt last
