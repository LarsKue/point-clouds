
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
            track_grad_norm=2,
            gradient_clip=None,
            profiler=None,
        )

    def __init__(self, train_data: Dataset = None, val_data: Dataset = None, test_data: Dataset = None, **hparams):
        super().__init__()
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def loss(self, batch, batch_idx) -> torch.Tensor:
        """ Compute the loss on the given batch """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("validation_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("test_loss", loss)

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

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_loss", save_last=True, every_n_epochs=25, save_top_k=5),
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
            track_grad_norm=self.hparams.track_grad_norm,
            profiler=self.hparams.profiler,
            benchmark=True,
        )
