import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Any
import torchvision
import wandb
import os


class EvalSaveCallback(pl.Callback):

    def __init__(self, save_dir: Path) -> None:
        super().__init__()
        self.save_dir = save_dir

    def save(self, outputs, batch, batch_idx):
        name = batch['name']

        filename = self.save_dir / f"{batch_idx:06d}_{name[0]}.pt"
        torch.save({
            "fpv": batch['image'],
            "seg_masks": batch['seg_masks'],
            'name': name,
            "output": outputs["output"],
            "valid_bev": outputs["valid_bev"],
        }, filename)

    def on_test_batch_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          outputs: torch.Tensor | Any | None,
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        if not outputs:
            return

        self.save(outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: torch.Tensor | Any | None,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if not outputs:

            return

        self.save(outputs, batch, batch_idx)


class ImageLoggerCallback(pl.Callback):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def log_image(self, trainer, pl_module, outputs, batch, batch_idx, mode="train"):
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return

        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return

        fpv_rgb = batch["image"]
        fpv_grid = torchvision.utils.make_grid(
            fpv_rgb, nrow=8, normalize=False)
        images = [
            wandb.Image(fpv_grid, caption="fpv")
        ]

        pred = outputs['output'].permute(0, 2, 3, 1)
        pred[outputs["valid_bev"][..., :-1] == 0] = 0
        pred = (pred > 0.5).float()
        pred = pred.permute(0, 3, 1, 2)

        for i in range(self.num_classes):
            gt_class_i = batch['seg_masks'][..., i]
            gt_class_i_grid = torchvision.utils.make_grid(
                gt_class_i.unsqueeze(1), nrow=8, normalize=False, pad_value=0)
            pred_class_i = pred[:, i]
            pred_class_i_grid = torchvision.utils.make_grid(
                pred_class_i.unsqueeze(1), nrow=8, normalize=False, pad_value=0)

            images += [
                wandb.Image(gt_class_i_grid, caption=f"gt_class_{i}"),
                wandb.Image(pred_class_i_grid, caption=f"pred_class_{i}")
            ]

        experiment.log(
            {
                "{}/images".format(mode): images
            }
        )

    def on_validation_batch_end(self, trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if batch_idx == 0:
            with torch.no_grad():
                outputs = pl_module(batch)
            self.log_image(trainer, pl_module, outputs,
                           batch, batch_idx, mode="val")

    def on_train_batch_end(self, trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if batch_idx == 0:
            pl_module.eval()

            with torch.no_grad():
                outputs = pl_module(batch)

            self.log_image(trainer, pl_module, outputs,
                           batch, batch_idx, mode="train")

            pl_module.train()


class KeepLastKCheckpoints(pl.Callback):
    """Keep only the last K checkpoint files in a directory (by mtime).

    This callback deletes older .ckpt files in `dirpath` leaving only the
    `keep_last_k` most recently modified files. Use this when you want to
    preserve the most recent checkpoints rather than the best ones.
    """

    def __init__(self, dirpath: Path | str, keep_last_k: int = 4) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.keep_last_k = int(keep_last_k)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Any) -> None:
        try:
            if not self.dirpath.exists():
                return

            files = [p for p in self.dirpath.glob("*.ckpt") if p.is_file()]
            if len(files) <= self.keep_last_k:
                return

            files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
            to_delete = files_sorted[self.keep_last_k:]
            for f in to_delete:
                try:
                    f.unlink()
                except Exception:
                    # ignore deletion errors
                    pass
        except Exception:
            # never raise from a callback
            return


class PeriodicCheckpointCallback(pl.Callback):
    def __init__(self, dirpath: Path | str, every_n_epochs: int = 50) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.every_n_epochs = int(every_n_epochs)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.every_n_epochs <= 0:
            return

        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        try:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            filename = self.dirpath / f"epoch={trainer.current_epoch + 1}-step={trainer.global_step}.ckpt"
            trainer.save_checkpoint(str(filename))
        except Exception:
            return
