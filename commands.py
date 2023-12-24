import os
from pathlib import Path

import fire
import lightning.pytorch as pl
import torch
from hydra import compose, initialize
from natsort import natsorted

from MNIST.fbp.data import MyDataModule
from MNIST.fbp.model import LightningMNISTClassifier


def train(cfg="./MNIST/conf/config.yaml"):
    cfg = Path(cfg)
    with initialize(
        version_base=None,
        config_path=str(cfg.parent),
        job_name="mnist_classifier",
    ):
        cfg = compose(config_name=cfg.stem)
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    dm = MyDataModule(
        val_size=cfg.data.val_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        batch_size=cfg.data.batch_size,
    )
    model = LightningMNISTClassifier(cfg)

    loggers = [
        # pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="file:./.logs/my-mlflow-logs",
        ),
        # pl.loggers.TensorBoardLogger(
        #     "./.logs/my-tb-logs", name=cfg.artifacts.experiment_name
        # ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]

    if cfg.callbacks.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=cfg.callbacks.swa.lrs)
        )

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(
                    cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name
                ),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_steps=cfg.trainer.num_warmup_steps + cfg.trainer.num_training_steps,
        accumulate_grad_batches=cfg.trainer.grad_accum_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        overfit_batches=cfg.trainer.overfit_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        deterministic=cfg.trainer.full_deterministic_mode,
        benchmark=cfg.trainer.benchmark,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        profiler=cfg.trainer.profiler,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        detect_anomaly=cfg.trainer.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    if cfg.trainer.batch_size_finder:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


def infer(cfg="./MNIST/conf/infer_config.yaml"):
    cfg = Path(cfg)
    with initialize(
        version_base=None,
        config_path=str(cfg.parent),
        job_name="mnist_classifier",
    ):
        cfg = compose(config_name=cfg.stem)
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    dm = MyDataModule(
        val_size=cfg.data.val_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        batch_size=cfg.data.batch_size,
    )

    dm.setup("validate")
    val_loader = dm.val_dataloader()

    model = LightningMNISTClassifier(cfg)
    ckpt_name = natsorted(os.listdir("checkpoints/example-experiment/"))[-1]
    ckpt_path = os.path.join("checkpoints", cfg.artifacts.experiment_name, ckpt_name)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    trainer.validate(model, dataloaders=val_loader)


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
