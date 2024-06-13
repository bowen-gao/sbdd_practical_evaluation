# python train.py \
#     experiment=lba30 \
#     scheduler.num_warmup_steps=200 \
#     logging.wandb.name=profsa_base_lba30 \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler.num_warmup_steps=200 \
#     logging.wandb.name=profsa_base_lba60 \
#     logging.wandb.tags="[lba, lba60]"

# python train.py \
#     experiment=lba30 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=20 \
#     logging.wandb.name=profsa_base_lba30_constant \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=20 \
#     logging.wandb.name=profsa_base_lba60_constant \
#     logging.wandb.tags="[lba, lba60]"

# python train.py \
#     experiment=lba30 \
#     scheduler.num_warmup_steps=200 \
#     dataset.batch_size=32 \
#     logging.wandb.name=profsa_base_lba30_bz32 \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler.num_warmup_steps=200 \
#     dataset.batch_size=32 \
#     logging.wandb.name=profsa_base_lba60_bz32 \
#     logging.wandb.tags="[lba, lba60]"

# python train.py \
#     experiment=lba30 \
#     scheduler.num_warmup_steps=200 \
#     trainer.precision=16 \
#     logging.wandb.name=profsa_base_lba30_fp16 \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler.num_warmup_steps=200 \
#     trainer.precision=16 \
#     logging.wandb.name=profsa_base_lba60_fp16 \
#     logging.wandb.tags="[lba, lba60]"


# python train.py \
#     experiment=lba30 \
#     scheduler.num_warmup_steps=200 \
#     optim.lr=0.0002 \
#     logging.wandb.name=profsa_base_lba30_lr2e-4 \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler.num_warmup_steps=200 \
#     optim.lr=0.0002 \
#     logging.wandb.name=profsa_base_lba60_lr2e-4 \
#     logging.wandb.tags="[lba, lba60]"

# python train.py \
#     experiment=lba30 \
#     scheduler.num_warmup_steps=200 \
#     optim.lr=0.00005 \
#     logging.wandb.name=profsa_base_lba30_lr5e-5 \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler.num_warmup_steps=200 \
#     optim.lr=0.00005 \
#     logging.wandb.name=profsa_base_lba60_lr5e-5 \
#     logging.wandb.tags="[lba, lba60]"

# python train.py \
#     experiment=lba30 \
#     scheduler.num_warmup_steps=200 \
#     model.cfg.pretrained_weights=/log/train/profsa/profsa_ckpt_modified.pt \
#     logging.wandb.name=profsa_base_lba30_ori \
#     logging.wandb.tags="[lba, lba30]"

# python train.py \
#     experiment=lba60 \
#     scheduler.num_warmup_steps=200 \
#     model.cfg.pretrained_weights=/log/train/profsa/profsa_ckpt_modified.pt \
#     logging.wandb.name=profsa_base_lba60_ori \
#     logging.wandb.tags="[lba, lba60]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.1 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.1 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.15 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.15 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.2 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.2 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.25 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.25 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.3 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.3 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.35 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.35 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.4 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.4 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.45 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.45 \
    logging.wandb.tags="[lba, lba30]"

python train.py \
    experiment=lba30 \
    model.cfg.pretrained_weights="/log/train/profsa2/ProFSADataModule.DrugCLIP.ScreeningCriterion.2024-01-30_13-51-47/checkpoints/epoch\\=064-step\\=370370.ckpt" \
    model.cfg.dropout=0.5 \
    optim.lr=0.0002 \
    logging.wandb.name=profsa_base_lba30_drop0.5 \
    logging.wandb.tags="[lba, lba30]"