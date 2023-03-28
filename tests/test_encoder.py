
import torch
import torch.nn as nn

from point_clouds.models import PointCloudsModel, PointCloudsModelHParams
from point_clouds.models.encoder.pools import GlobalMultimaxPool1d


def test_permutation_invariance(encoder):
    equivariant_module = nn.Sequential(
        nn.Linear(3, 128),
        GlobalMultimaxPool1d(64, dim=1)
    )
    encoder_hparams = dict(
        equivariant_module=equivariant_module,

    )

    hparams = PointCloudsModelHParams(
        inputs=3,
        points=2048,
        conditions=128,

        encoder_hparams=encoder_hparams,
        rectifier_hparams=rectifier_hparams,

        max_epochs=1,
        batch_size=4,
    )

    model = PointCloudsModel(hparams)

    permutation = torch.randperm(hparams.points)
    x = torch.randn(hparams.batch_size, hparams.points, hparams.inputs)
    xp = x[:, permutation, :]

    torch.manual_seed(42)
    z = encoder(x)

    torch.manual_seed(42)
    zp = encoder(xp)

    assert torch.allclose(z, zp, atol=1e-7)
