# python train.py experiment=base \
#     trainer.max_epochs=3 \
#     scheduler.num_warmup_steps=1000 \
#     logging.wandb.name=base_warmup_1000

# python train.py experiment=base \
#     trainer.max_epochs=3 \
#     scheduler.num_warmup_steps=500 \
#     logging.wandb.name=base_warmup_500

# python train.py experiment=base \
#     trainer.max_epochs=3 \
#     optim.lr=0.0002 \
#     logging.wandb.name=base_lr_0.0002

# python train.py experiment=base \
#     trainer.max_epochs=3 \
#     optim.lr=0.005 \
#     logging.wandb.name=base_lr_0.0004

# python train.py experiment=base \
#     trainer.max_epochs=3 \
#     model.cfg.logit_scale=1.0 \
#     logging.wandb.name=base_logit_scale_1

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=2.0 \
#     logging.wandb.tags="[logit_scale]" \
#     logging.wandb.name=base_logit_scale_2

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=4.0 \
#     logging.wandb.tags="[logit_scale]" \
#     logging.wandb.name=base_logit_scale_4

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=8.0 \
#     logging.wandb.tags="[logit_scale]" \
#     logging.wandb.name=base_logit_scale_8

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=16.0 \
#     logging.wandb.tags="[logit_scale]" \
#     logging.wandb.name=base_logit_scale_16

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=32.0 \
#     logging.wandb.tags="[logit_scale]" \
#     logging.wandb.name=base_logit_scale_32

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=64.0 \
#     logging.wandb.tags="[logit_scale]" \
#     logging.wandb.name=base_logit_scale_64

# python train.py experiment=base_find_hparam \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_find_hparam

# python train.py experiment=base_find_hparam \
#     optim.lr=0.0002 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_lr_0.0002

# python train.py experiment=base_find_hparam \
#     optim.lr=0.0004 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_lr_0.0004

# python train.py experiment=base_find_hparam \
#     optim=adamw \
#     optim.lr=0.0001 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_adamw

# python train.py experiment=base_find_hparam \
#     optim=adamw \
#     optim.lr=0.0002 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_adamw_lr_0.0002

# python train.py experiment=base_find_hparam \
#     optim=adamw \
#     optim.lr=0.0004 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_adamw_lr_0.0004

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=16.0 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_logit_scale_16

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=32.0 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_logit_scale_32

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=32.0 \
#     optim.lr=0.0002 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_logit_scale_32_lr_0.0002

# python train.py experiment=base_find_hparam \
#     model.cfg.logit_scale=16.0 \
#     optim.lr=0.0002 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_logit_scale_16_lr_0.0002

# python train.py experiment=base_find_hparam \
#     seed=3407 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_seed_3407

# python train.py experiment=base_find_hparam \
#     seed=2024 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_seed_2024

# python train.py experiment=base_find_hparam \
#     seed=1 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_seed_1

# python train.py experiment=base_find_hparam \
#     seed=1 \
#     model.cfg.logit_scale=32.0 \
#     logging.wandb.tags="[find_hparam]" \
#     logging.wandb.name=base_seed_1_logit_scale_32

# python train.py experiment=base \
#     seed=1 \
#     logging.wandb.tags="[profsa]" \
#     logging.wandb.name=profsa_seed_1

# batch size tuning

python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=16 \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_bz16 \
    trainer.devices="[0,1,2,3]"

python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=32 \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_bz32 \
    trainer.devices="[0,1,2,3]"

python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=64 \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_bz64 \
    trainer.devices="[4,5,6,7]"

python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=80 \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_bz80 \
    trainer.devices="[0,1,2,3]"

python train.py experiment=base \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=112 \
    optim=adamw \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_bz112_adamw \
    trainer.devices="[4,5,6,7]"

# dropout

python train.py experiment=base_dropout \
    seed=1 \
    trainer.precision="16-mixed" \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_dropout \
    trainer.devices="[4,5,6,7]"

python train.py experiment=base_dropout \
    seed=1 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=112 \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_dropout_bz112 \
    trainer.devices="[4,5,6,7]"

# more warmup steps
python train.py experiment=base_dropout \
    seed=2024 \
    trainer.precision="16-mixed" \
    dataset.batch_size.train=112 \
    scheduler.num_warmup_steps=20000 \
    logging.wandb.tags="[profsa, bz]" \
    logging.wandb.name=profsa_fp16_4gpu_dropout_bz112_seed2024_warmup2w
