
import pytorch_lightning as lightning

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from distributions import StandardNormal

from scipy.integrate import RK45

import utils






class RectifyingFlow(lightning.LightningModule):
    def __init__(self, train_data=None, val_data=None, test_data=None, **hparams):
        super().__init__()
        # merge hparams with merge operator (python 3.9+)
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams, ignore=["train_data", "val_data", "test_data"])

        self.integrator = EulerIntegrator()
        self.distribution = self.configure_distribution()
        self.net = self.configure_subnet()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        # self.train_data = None
        # if train_data is not None:
        #     latent_data = self.distribution.sample((len(train_data),))
        #     latent_data = SingleTensorDataset(latent_data)
        #     self.train_data = PairedDataset(train_data, latent_data)
        #
        # self.val_data = None
        # if val_data is not None:
        #     latent_data = self.distribution.sample((len(val_data),))
        #     latent_data = SingleTensorDataset(latent_data)
        #     self.val_data = PairedDataset(val_data, latent_data)
        #
        # self.test_data = None
        # if test_data is not None:
        #     latent_data = self.distribution.sample((len(test_data),))
        #     latent_data = SingleTensorDataset(latent_data)
        #     self.test_data = PairedDataset(test_data, latent_data)

        # self.train_data = None
        # if train_data is not None:
        #     latent_data = self.distribution.sample((len(train_data),))
        #     latent_data = SingleTensorDataset(latent_data)
        #     product_set = ProductSet(train_data, latent_data)
        #     self.train_data = ProductSet(train_data, latent_data)
        #
        # self.val_data = None
        # if val_data is not None:
        #     latent_data = self.distribution.sample((len(val_data),))
        #     latent_data = SingleTensorDataset(latent_data)
        #     self.val_data = ProductSet(val_data, latent_data)
        #
        # self.test_data = None
        # if test_data is not None:
        #     latent_data = self.distribution.sample((len(test_data),))
        #     latent_data = SingleTensorDataset(latent_data)
        #     self.test_data = ProductSet(test_data, latent_data)

    def resample(self):
        train_data = self.train_data.datasets[0]
        latent_data = self.forward_inference(train_data.tensors[0])
        latent_data = SingleTensorDataset(latent_data)

        self.train_data = PairedDataset(train_data, latent_data)

    @property
    def default_hparams(self):
        return dict(
            batch_size=1,
            time_samples=1,
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            lr_warmup_milestones=[],
            lr_warmup_gamma=10.0,
            lr_milestones=[],
            lr_gamma=0.1,
            integration_method="euler",
            integration_steps=100,
        )

    def forward_inference(self, x: torch.Tensor):
        t0 = x.new_zeros(x.shape[0], 1)
        dt = 1.0 / self.hparams.integration_steps
        z = self.integrator.solve(x, self.net, t0=t0, dt=dt)
        return z

    def inverse_inference(self, z: torch.Tensor):
        def f(x, t):
            return -self.net(x, time=1 - t)

        t0 = z.new_zeros(z.shape[0], 1)
        dt = 1.0 / self.hparams.integration_steps
        x = self.integrator.solve(z, f, t0=t0, dt=dt)

        # t0 = z.new_ones(z.shape[0], 1)
        # dt = -1.0 / self.hparams.integration_steps
        # x = self.integrator.solve(z, self.net, t0=t0, dt=dt)

        return x

    def forward(self, batch, batch_idx):
        x0 = batch
        batch_size = x0.shape[0]

        x1 = self.distribution.sample((batch_size,)).to(self.device)

        t = torch.rand(*x0.shape, self.hparams.time_samples).to(self.device)

        xt = t * x1[..., None] + (1 - t) * x0[..., None]
        xt = xt.movedim(-1, 1).flatten(start_dim=0, end_dim=1)

        predicted = self.net(xt, time=t)
        true = (x1 - x0).repeat(self.hparams.time_samples, 1)

        return predicted, true

    def training_step(self, batch, batch_idx):
        predicted, true = self.forward(batch, batch_idx)

        residuals = torch.abs(predicted - true)

        mse = utils.sum_except_batch(torch.square(residuals)).mean(dim=0)
        mean_prediction_error = utils.sum_except_batch(predicted - true).mean(dim=0)

        self.log("training_mse", mse)
        self.log("training_mpe", mean_prediction_error)

        return mse

    def validation_step(self, batch, batch_idx):
        predicted, true = self.forward(batch, batch_idx)

        residuals = torch.abs(predicted - true)

        mse = utils.sum_except_batch(torch.square(residuals)).mean(dim=0)
        mean_prediction_error = utils.sum_except_batch(predicted - true).mean(dim=0)

        self.log("validation_mse", mse)
        self.log("validation_mpe", mean_prediction_error)

    # def test_step(self, batch, batch_idx):
    #     loss = self._step(batch, batch_idx)
    #     self.log("test_loss", loss)

    def generate(self, shape=torch.Size((1,)), temperature=1.0):
        """ Generate a number of samples by sampling randomly from the latent distribution """
        z = temperature * self.distribution.sample(shape).to(self.device)

        return self.inverse_inference(z)

    def configure_optimizers(self):
        """
        Configure optimizers and LR schedulers to be used in training
        :return: Dict containing the configuration
        """
        match self.hparams.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            case _:
                raise ValueError(f"Unsupported Optimizer: {self.hparams.optimizer}")

        lr_warmup = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.lr_warmup_milestones,
            gamma=self.hparams.lr_warmup_gamma
        )
        lr_step = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.lr_milestones,
            gamma=self.hparams.lr_gamma
        )
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            lr_warmup,
            lr_step,
        ])

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_mse", save_last=True),
            lightning.callbacks.LearningRateMonitor(),
        ]

    def configure_inn(self):
        """
        Configure and return the inn used by this module
        """
        raise NotImplementedError

    def configure_subnet(self):
        """
        Configure and return the subnetwork used by individual inn modules to predict parameters
        """
        return Subnet(self.hparams.input_shape[0])

    def configure_distribution(self):
        """
        Configure and return the latent distribution used by this module
        """
        return StandardNormal(self.hparams.input_shape)

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


class Subnet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(in_features=dim + 1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=dim)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = torch.cat((time, x), dim=-1)
        return self.net.forward(x)


from torch.utils.data import Dataset








class ProductSet(Dataset):
    def __init__(self, d1: Dataset, d2: Dataset):
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1) * len(self.d2)

    def __getitem__(self, item):
        i, j = item % len(self.d1), item // len(self.d1)
        return self.d1[i], self.d2[j]



from typing import Callable, Generic, TypeVar, Union


T = TypeVar("T")


class Integrator(Generic[T]):
    def step(self, x: T, f: Callable[[T, float], T], t: Union[float, T], dt: float) -> T:
        raise NotImplementedError

    def solve(self, x: T, f: Callable[[T, float], T], t0: float, dt: float, steps: int = 1) -> T:
        t = t0
        for step in range(steps):
            x = self.step(x, f, t, dt)
            t = t + dt

        return x


class EulerIntegrator(Integrator):
    def step(self, x: T, f: Callable[[T, float], T], t: float, dt: float):
        return x + dt * f(x, t)


class RungeKuttaIntegrator(Integrator):
    def solve(self, x: T, f: Callable[[T, float], T], t0: float, dt: float, steps: int = 1) -> T:
        rk45 = RK45(f, t0, x, t_bound=t0 + steps * dt)



