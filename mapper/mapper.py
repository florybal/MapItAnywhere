import time
import torch
import hydra
import logging
import sys
import pytorch_lightning as pl
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pathlib import Path
from dataclasses import dataclass

from .module import GenericModule
from .data.module import GenericDataModule
from .callbacks import EvalSaveCallback, ImageLoggerCallback
from .models.schema import ModelConfiguration, DINOConfiguration, ResNetConfiguration
from .data.schema import MIADataConfiguration, KITTIDataConfiguration, NuScenesDataConfiguration, IndoorDataConfiguration


@dataclass
class ExperimentConfiguration:
    name: str

@dataclass
class Configuration:
    model: ModelConfiguration
    experiment: ExperimentConfiguration
    data: Any
    training: Any


cs = ConfigStore.instance()

# Store root configuration schema
cs.store(name="pretrain", node=Configuration)
cs.store(name="mapper_nuscenes", node=Configuration)
cs.store(name="mapper_kitti", node=Configuration)

# Store data configuration schema
cs.store(group="schema/data", name="mia",
         node=MIADataConfiguration, package="data")
cs.store(group="schema/data", name="kitti", node=KITTIDataConfiguration, package="data")
cs.store(group="schema/data", name="nuscenes", node=NuScenesDataConfiguration, package="data")
cs.store(group="schema/data", name="indoor", node=IndoorDataConfiguration, package="data")

# Configuração principal para indoor
cs.store(name="mapper_indoor", node=Configuration)

cs.store(group="model/schema/backbone", name="dino", node=DINOConfiguration, package="model.image_encoder.backbone")
cs.store(group="model/schema/backbone", name="resnet", node=ResNetConfiguration, package="model.image_encoder.backbone")


@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def train(cfg: Configuration):
    OmegaConf.resolve(cfg)

    # reduce verbosity from dataset and other modules by default
    logging.getLogger("mapper").setLevel(logging.WARNING)
    logging.getLogger("mapper.data.indoor").setLevel(logging.WARNING)

    # use a dedicated python logger for informational messages; the variable
    # `pl_logger` or `wandb_logger` below is reserved for the PyTorch-Lightning
    # logger object which doesn't implement .info()/.error().
    py_logger = logging.getLogger("mapper")
    py_logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    dm = GenericDataModule(cfg.data)

    model = GenericModule(cfg)

    exp_name_with_time = cfg.experiment.name + \
        "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

    callbacks: list[pl.Callback]

    pl_logger = None
    if cfg.training.eval:
        save_dir = Path(cfg.training.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EvalSaveCallback(save_dir=save_dir)
        ]
    else:
        callbacks = [
            ImageLoggerCallback(num_classes=cfg.training.num_classes),
            ModelCheckpoint(
                monitor=cfg.training.checkpointing.monitor,
                save_last=cfg.training.checkpointing.save_last,
                save_top_k=cfg.training.checkpointing.save_top_k,
            )
        ]

        # prefer offline or disabled wandb if API key not present
        try:
            pl_logger = WandbLogger(
                name=exp_name_with_time,
                id=exp_name_with_time,
                entity="mappred-large",
                project="map-pred-full-v3",
            )
            # log graph only if wandb successfully initialises
            pl_logger.watch(model, log="all", log_freq=500)
            py_logger.info("Using WandbLogger %s", exp_name_with_time)
        except Exception as e:
            py_logger.warning(
                "WandbLogger unavailable (%s), continuing without it.\n"
                "To log metrics install/configure wandb or set WANDB_MODE=offline",
                str(e),
            )
            pl_logger = None

    if cfg.training.checkpoint is not None:
        # load checkpoint and drop any parameters whose shape doesn't match the
        # current model.  This allows fine‑tuning when the pretrained weights
        # come from a different class count or scale bin configuration.
        checkpoint = torch.load(cfg.training.checkpoint, weights_only=False)
        state_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        # identify mismatched keys
        bad_keys = []
        for k, v in list(state_dict.items()):
            if k in model_dict and v.shape != model_dict[k].shape:
                bad_keys.append(k)
                state_dict.pop(k)
        if bad_keys:
            py_logger = logging.getLogger("mapper")
            py_logger.warning(
                "Dropped %d incompatible keys from checkpoint: %s",
                len(bad_keys), bad_keys,
            )

        model.load_state_dict(state_dict, strict=False)

    trainer_args = OmegaConf.to_container(cfg.training.trainer)
    trainer_args['callbacks'] = callbacks
    trainer_args['logger'] = pl_logger

    trainer = pl.Trainer(**trainer_args)

    try:
        if cfg.training.eval:
            py_logger.info("Starting evaluation")
            trainer.test(model, datamodule=dm)
        else:
            py_logger.info("Starting training")
            trainer.fit(model, datamodule=dm)
    except Exception as exc:
        # catch and log any Python exception; downstream crashes (OOM, kernel panic)
        # will still bring the machine down but we will at least have a trace
        py_logger.error("Exception raised during trainer execution", exc_info=True)
        # flush stdout/stderr to make sure log file is written before crash
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    train()

