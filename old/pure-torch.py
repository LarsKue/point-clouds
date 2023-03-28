
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import make_moons

from distributions import StandardNormal

import numpy as np
import matplotlib.pyplot as plt


torch.autograd.set_grad_enabled(False)

plt.rc("figure", dpi=250)
plt.rc("legend", fontsize=6)


shape = (2,)
batch_size = 128

n_train = 128 * batch_size
n_val = 32 * batch_size


epochs = 100
lr = 1e-3
weight_decay = 1e-5

integration_steps = 10000


n = n_train + n_val

x0, _y = make_moons(n, shuffle=True, noise=0.05)
x0 = x0.astype(np.float32)
x0 = torch.from_numpy(x0)

distribution = StandardNormal(shape)

x1 = distribution.sample((n,))

train_data = TensorDataset(x0[:n_train], x1[:n_train])
val_data = TensorDataset(x0[n_train:n_train + n_val], x1[n_train:n_train + n_val])

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

model = nn.Sequential(
    nn.Linear(in_features=2, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=2),
)

model = model.cuda()

mse = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


for epoch in range(epochs):
    print("=" * 50)
    print(f"{epoch=}")

    with torch.enable_grad():
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            x0, x1 = batch
            x0 = x0.cuda()
            x1 = x1.cuda()

            x1 = distribution.sample((x0.shape[0],)).cuda()

            t = torch.rand(*x0.shape)
            t = t.cuda()

            xt = t * x1 + (1 - t) * x0

            predicted = model(xt)
            true = x1 - x0

            loss = mse(predicted, true)
            print(f"\rtrain loss: {loss:.2f}", end="")

            loss.backward()
            optimizer.step()

    print()

    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            x0, x1 = batch
            x0 = x0.cuda()
            x1 = x1.cuda()

            x1 = distribution.sample((x0.shape[0],)).cuda()

            t = torch.rand(*x0.shape)
            t = t.cuda()

            xt = t * x1 + (1 - t) * x0

            predicted = model(xt)
            true = x1 - x0

            loss = mse(predicted, true)
            print(f"\rvalidation loss: {loss:.2f}", end="")

    print()


# generate samples
x = distribution.sample((n_val,)).cuda()


for step in range(integration_steps):
    v = model(x)
    dt = 1 / integration_steps
    x = x - dt * v

x0 = val_data.tensors[0].cpu()
x = x.cpu()

plt.scatter(x0[:, 0], x0[:, 1], s=1, label="True")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.scatter(x[:, 0], x[:, 1], s=1, label="Sampled")
plt.legend()
plt.show()

