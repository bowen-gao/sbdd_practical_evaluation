export CUDA_VISIBLE_DEVICES="4,5,6,7"

# base
logdir='/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-08_21-48-13/checkpoints/epoch=076-step=438746.ckpt'
# profsa ori
logdir='/log/train/profsa/profsa_ckpt_modified.pt'
# seed = 1
logdir='/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-20_13-30-11/checkpoints/epoch=061-step=353276.ckpt'
# remove used parameters
logdir='/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-17_18-59-09/checkpoints/epoch=076-step=438746.ckpt'
# fp16 4gpu
logdir='/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch=064-step=370370.ckpt'
# new split (19)
logdir=/log/train/profsa/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-02-29_13-48-07/checkpoints/epoch_099_513100.ckpt

logdir=${logdir//=/\\=}


python train.py \
    experiment=lba30 \
    scheduler.num_warmup_steps=200 \
    optim.lr=0.0002 \
    model.cfg.dropout=0.5 \
    model.cfg.pretrained_weights=$logdir \
    logging.wandb.name=profsa_base_lba30 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba60 \
    scheduler.num_warmup_steps=200 \
    optim.lr=0.0002 \
    model.cfg.dropout=0 \
    model.cfg.pretrained_weights=$logdir \
    logging.wandb.name=profsa_base_lba60 \
    logging.wandb.tags="[lba, lba60]"
