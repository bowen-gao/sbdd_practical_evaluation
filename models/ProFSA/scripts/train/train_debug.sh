# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     trainer.precision=16 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.01 \
#     logging.wandb.name=debug_lr_0.01_fp16

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     trainer.precision=16 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.001 \
#     logging.wandb.name=debug_lr_0.001_fp16

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     trainer.precision=16 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.0001 \
#     logging.wandb.name=debug_lr_0.0001_fp16

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     trainer.precision=16 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.00001 \
#     logging.wandb.name=debug_lr_0.00001_fp16

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.1 \
#     logging.wandb.name=debug_lr_0.1

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.01 \
#     logging.wandb.name=debug_lr_0.01

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.001 \
#     logging.wandb.name=debug_lr_0.001

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.0001 \
#     logging.wandb.name=debug_lr_0.0001

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.00001 \
#     logging.wandb.name=debug_lr_0.00001

# python train.py \
#     +trainer.max_steps=8000 \
#     +trainer.track_grad_norm=2 \
#     scheduler=constant \
#     scheduler.num_warmup_steps=100 \
#     optim.lr=0.000001 \
#     logging.wandb.name=debug_lr_0.000001

python train.py \
    +trainer.max_steps=8000 \
    +trainer.track_grad_norm=2 \
    scheduler=polynomial_decay \
    optim.lr=0.0001 \
    logging.wandb.name=debug_polynomial_decay

python train.py \
    +trainer.max_steps=8000 \
    +trainer.track_grad_norm=2 \
    trainer.precision=bf16 \
    scheduler=constant \
    scheduler.num_warmup_steps=100 \
    optim.lr=0.0001 \
    logging.wandb.name=debug_lr_0.0001_bf16

python train.py \
    +trainer.max_steps=8000 \
    trainer.precision="16-mixed" \
    logging.wandb.name=debug_16_mixed
