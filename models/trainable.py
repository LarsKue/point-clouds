
import pytorch_lightning as lightning
import torch

from torch.utils.data import DataLoader, Dataset


class Trainable(lightning.LightningModule):

    @property
    def default_hparams(self):
        return dict(
            accelerator="gpu",
            devices=1,
            max_epochs=None,
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=1,
            accumulate_batches=None,
            gradient_clip=None,
        )

    def __init__(self, train_data: Dataset = None, val_data: Dataset = None, test_data: Dataset = None, **hparams):
        super().__init__()
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def configure_optimizers(self):
        """
        Configure optimizers and LR schedulers for Lightning
        """
        lr = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay
        match self.hparams.optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
            case optimizer:
                raise NotImplementedError(f"Unsupported Optimizer: {optimizer}")

        return optimizer

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_loss", save_last=True),
            lightning.callbacks.LearningRateMonitor(),
        ]

    def train_dataloader(self):
        """
        Configure and return the train dataloader
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        """
        Configure and return the validation dataloader
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        """
        Configure and return the test dataloader
        """
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def configure_trainer(self):
        """
        Configure and return the Trainer used to train this module
        """
        return lightning.Trainer(
            accelerator=self.hparams.accelerator.lower(),
            devices=self.hparams.devices,
            max_epochs=self.hparams.max_epochs,
            gradient_clip_val=self.hparams.gradient_clip,
            accumulate_grad_batches=self.hparams.accumulate_batches,
            track_grad_norm=2,
            benchmark=True,
        )
