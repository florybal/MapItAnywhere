import time
import os
import torch
import hydra
import pytorch_lightning as pl
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path
from dataclasses import dataclass

from .module import GenericModule
from .data.module import GenericDataModule
from .callbacks import EvalSaveCallback, ImageLoggerCallback, KeepLastKCheckpoints, PeriodicCheckpointCallback
from .models.schema import ModelConfiguration, DINOConfiguration, ResNetConfiguration
from .data.schema import MIADataConfiguration, KITTIDataConfiguration, NuScenesDataConfiguration


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

cs.store(group="model/schema/backbone", name="dino", node=DINOConfiguration, package="model.image_encoder.backbone")
cs.store(group="model/schema/backbone", name="resnet", node=ResNetConfiguration, package="model.image_encoder.backbone")


@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def train(cfg: Configuration):
    OmegaConf.resolve(cfg)

    # PyTorch 2.6 defaults to weights_only=True when Lightning resumes from a
    # checkpoint. Older Lightning checkpoints can store OmegaConf objects in the
    # serialized state, so allowlist the type before loading.
    try:
        torch.serialization.add_safe_globals([DictConfig, ContainerMetadata])
    except Exception:
        pass

    dm = GenericDataModule(cfg.data)

    model = GenericModule(cfg)

    exp_name_with_time = cfg.experiment.name + \
        "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

    callbacks: list[pl.Callback]

    if cfg.training.eval:
        save_dir = Path(cfg.training.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EvalSaveCallback(save_dir=save_dir)
        ]

        logger = None
    else:
        checkpointing_cfg = cfg.training.checkpointing
        every_n_epochs = None
        try:
            every_n_epochs = checkpointing_cfg.every_n_epochs
        except Exception:
            every_n_epochs = None

        monitor = checkpointing_cfg.monitor
        save_top_k = checkpointing_cfg.save_top_k
        if every_n_epochs is not None and every_n_epochs > 0:
            # Keep ModelCheckpoint dedicated to last.ckpt so resuming does not
            # inherit the old metric-ranked schedule from the checkpoint state.
            monitor = None
            save_top_k = 0

        callbacks = [
            ImageLoggerCallback(num_classes=cfg.training.num_classes),
            ModelCheckpoint(
                monitor=monitor,
                save_last=checkpointing_cfg.save_last,
                save_top_k=save_top_k,
            )
        ]

        if every_n_epochs is not None and every_n_epochs > 0:
            callbacks.append(
                PeriodicCheckpointCallback(
                    dirpath=cfg.training.checkpointing.dirpath,
                    every_n_epochs=every_n_epochs,
                )
            )

        # Optionally add a callback to keep only the last K checkpoint files
        # (useful when you want the most recent N checkpoints instead of the
        # best ones by validation metric).
        keep_last_k = None
        try:
            keep_last_k = cfg.training.checkpointing.keep_last_k
        except Exception:
            keep_last_k = None

        if keep_last_k is not None:
            callbacks.append(
                KeepLastKCheckpoints(dirpath=cfg.training.checkpointing.dirpath, keep_last_k=keep_last_k)
            )
        
        # Early stopping: prevent overfitting and stop when validation loss doesn't improve
        if cfg.training.early_stopping.enabled:
            callbacks.append(
                EarlyStopping(
                    monitor=cfg.training.early_stopping.monitor,
                    patience=cfg.training.early_stopping.patience,
                    min_delta=cfg.training.early_stopping.min_delta,
                    mode=cfg.training.early_stopping.mode,
                    verbose=True,
                )
            )

        logger = WandbLogger(
            name=exp_name_with_time,
            id=exp_name_with_time,
            entity="mappred-large",
            project="map-pred-full-v3",
            mode="offline",
        )

        try:
            logger.watch(model, log="all", log_freq=500)
        except Exception as e:
            print(f"Warning: Could not setup WandB watch: {e}")

    if cfg.training.checkpoint is not None:
        ckpt = torch.load(cfg.training.checkpoint, weights_only=False)['state_dict']
        # Filter out keys with mismatched shapes (e.g., final segmentation head when num_classes differ)
        model_state = model.state_dict()
        filtered_ckpt = {
            k: v
            for k, v in ckpt.items()
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered_ckpt, strict=False)
        if missing_keys:
            print(f"Warning: missing keys after loading checkpoint: {missing_keys}")

    trainer_args = OmegaConf.to_container(cfg.training.trainer)
    trainer_args['callbacks'] = callbacks
    trainer_args['logger'] = logger

    trainer = pl.Trainer(**trainer_args)

    if cfg.training.eval:
        trainer.test(model, datamodule=dm)
    else:
        ckpt_path = None
        try:
            ckpt_path = cfg.training.resume_from_checkpoint
        except Exception:
            ckpt_path = None

        if ckpt_path is not None:
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    train()

