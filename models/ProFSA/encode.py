import argparse
import copy
import logging
import os
import sys
import time
import hydra
from pathlib import Path
from typing import Callable, Iterable, List, Union

import lightning as lt
import wandb
from hydra import compose, initialize, initialize_config_dir
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf, open_dict, DictConfig

from src.utils.exptool import (
    Experiment,
    prepare_trainer_config,
    print_config,
    register_omegaconf_resolver,
)

register_omegaconf_resolver()

logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

main_dir = Path(__file__).resolve().parent



def encode(
    cfg: DictConfig,
):
    logdir = Path(cfg.log_dir).expanduser()
    os.chdir(logdir)

    # load experiment record from logdir
    experiment = Experiment(logdir)

    exp_config = copy.deepcopy(experiment.config)
    # OmegaConf.set_struct(exp_config, True)

    ############################################################
    # here to modify how to merge the exp_config to the new cfg
    ############################################################

    cfg_trainer = dict(cfg.trainer)
    with open_dict(cfg):
        # cfg.dataset = exp_config.dataset
        if "dataset" not in cfg:
            cfg.dataset = exp_config.dataset
        cfg.seed = exp_config.seed
        cfg.model = exp_config.model
        cfg.trainer = exp_config.trainer
    
        for key, val in cfg_trainer.items():
            cfg.trainer[key] = val

        # cfg.dataset.dataset_cfg.train.data_file="train.lmdb"

    ############################################################
    # end of modification
    ############################################################    
    
    # show experiment config
    print_config(exp_config)
    print_config(cfg)

    # seed everything
    lt.seed_everything(cfg.seed)

    # initialize datamodule
    datamodule = instantiate(cfg.dataset)

    # initialize model
    pipeline = experiment.get_pipeline_model_loaded(cfg.ckpt)

    # initialize trainer
    cfg_trainer = prepare_trainer_config(cfg, logging=False)
    trainer = instantiate(cfg_trainer)


    # predict 
    results = trainer.predict(pipeline, datamodule=datamodule)



@hydra.main(version_base="1.3", config_path="conf", config_name="encode")
def main(cfg: DictConfig) -> None:
    print_config(cfg)
    encode(cfg)
    exit()

if __name__ == "__main__":
    main()
