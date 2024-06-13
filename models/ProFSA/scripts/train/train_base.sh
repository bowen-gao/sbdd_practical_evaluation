python train.py experiment=base

python train.py experiment=base

python train.py experiment=base \
    model.cfg.logit_scale=32.0 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_base_logit_scale_32

python train.py experiment=base \
    seed=1 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_seed_1

python train.py experiment=base \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=96 \
    dataset.batch_size.train=128 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16 \
    trainer.devices="[2]"

# fp16 4gpu
python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16_4gpu \
    trainer.devices="[4,5,6,7]"

# fp16 4gpu train batch = 96
python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size=96 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16_4gpu_bz96


# fp16 4gpu train batch = 96
python train.py experiment=base \
    seed=1 \
    optim.lr=0.0002 \
    trainer.precision="16-mixed" \
    dataset.batch_size=96 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16_4gpu_bz96_lrx2

# fp16 8gpu train batch = 96
# unstable
python train.py experiment=base \
    seed=1 \
    optim.lr=0.0002 \
    trainer.devices=8 \
    trainer.precision="16-mixed" \
    dataset.batch_size=112 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16_8gpu_bz112_lrx2


python train.py experiment=base \
    seed=1 \
    trainer.devices=8 \
    trainer.precision="16-mixed" \
    dataset.batch_size=112 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_fp16_8gpu_bz112

# tune pocket
python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    model.cfg.mol.fixed=false \
    logging.wandb.tags="[profsa, tune]" \
    logging.wandb.name=profsa_tune_mol \
    trainer.devices="[4,5,6,7]"

python train.py experiment=base_tune \
    model.cfg.mol.fixed=false \
    model.cfg.pocket.fixed=false \
    logging.wandb.tags="[profsa, tune]" \
    logging.wandb.name=profsa_tune_both

python train.py experiment=base_tune \
    model.cfg.mol.fixed=false \
    model.cfg.pocket.fixed=true \
    logging.wandb.tags="[profsa, tune]" \
    logging.wandb.name=profsa_tune_mol_fix_pocket \
    trainer.devices="[4,5,6,7]"


# fp16 4gpu
python train.py experiment=base_profsa2 \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa2_base \
    trainer.devices="[4,5,6,7]"

# fp16 4gpu new data split
python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.dataset_cfg.train.data_dir=/data/prot_frag/profsa \
    dataset.dataset_cfg.train.data_file=train.lmdb \
    model.cfg.data_dir=/data/prot_frag/profsa \
    logging.wandb.tags="[profsa]" \
    logging.wandb.name=profsa_base_new_split
