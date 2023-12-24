import lightning.pytorch as pl
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F


# from torch.utils.data import TensorDataset, DataLoader, random_split
# from torchvision.datasets import MNIST
# from torchvision import datasets, transforms

# from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger, ModelCheckpoint


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # Define the linear layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)  # Assuming input image size of 28x28
        self.fc2 = nn.Linear(256, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.lr_rate = cfg.trainer.opt_cfg.lr
        self.scheduler = cfg.trainer.scheduler.name

        self.training_step_outputs = (
            []
        )  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = (
            []
        )  # save targets in each batch to compute metric overall epoch
        self.val_step_outputs = (
            []
        )  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []

    def forward(self, x):
        # Apply the convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the output for the linear layers
        x = x.view(x.size(0), -1)

        # Apply the linear layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # probability distribution over labels
        x = self.logsoftmax(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {"train_loss": loss}
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        self.training_step_outputs.extend(y_pred)
        self.training_step_targets.extend(y_true)

        return {"loss": loss, "log": logs}

    def on_train_epoch_end(self):
        # F1 Macro all epoch saving outputs and target per batch
        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets
        f1_macro_epoch = f1_score(train_all_outputs, train_all_targets, average="macro")
        self.log(
            "training_f1_epoch",
            f1_macro_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # free up the memory
        # --> HERE STEP 3 <--
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        # Train loss per batch in epoch
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        # --> HERE STEP 2 <--
        self.val_step_outputs.extend(y_pred)
        self.val_step_targets.extend(y_true)

        return {"val_loss": loss}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"test_loss": loss}

    def on_validation_epoch_end(self):
        # F1 Macro all epoch saving outputs and target per batch
        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets
        val_f1_macro_epoch = f1_score(val_all_outputs, val_all_targets, average="macro")
        self.log(
            "val_f1_epoch",
            val_f1_macro_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # free up the memory
        # --> HERE STEP 3 <--
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def on_test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        if self.scheduler:
            if self.scheduler == "expo_lr":
                lr_scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=0.95
                    ),
                    "name": self.scheduler,
                }
            else:
                raise NotImplementedError()

        return [optimizer], [lr_scheduler] if self.scheduler else [optimizer]
