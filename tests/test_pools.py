
import torch
import torch.nn as nn


def test_attention():
    from models.pools import ISAB, PMA
    from models.pools import MAB

    # TODO: test softmax in MHA
    # TODO: rectified flow fuzzy matching

    inputs = 3
    points = 2048
    batch_size = 32

    in_shape = (batch_size, points, inputs)

    network = nn.Sequential(
        nn.Linear(inputs, 128),
        # ISAB(128, 256, num_heads=4, inducing_points=32),
        # ISAB(256, 512, num_heads=5, inducing_points=32),
        # PMA(512, num_heads=6, num_seeds=32),
    )

    x = torch.randn()

    x = torch.randn(in_shape)
    z = network(x)

    print(z.shape)

    assert False

    # network = nn.Sequential(
    #     # (batch, 2048, 3)
    #     nn.Linear(inputs, 128),
    #     nn.ReLU(),
    #     # (batch, 2048, 128)
    #     ISAB(128, 128, num_heads=4, inducing_points=16),
    #     ISAB(128, 128, num_heads=4, inducing_points=16),
    #     nn.Dropout(dropout),
    #     PMA(128, num_heads=4, num_seeds=16),
    #     nn.Linear(128, 256),
    #     nn.ReLU(),
    #     ISAB(256, 256, num_heads=4, inducing_points=32),
    #     ISAB(256, 256, num_heads=4, inducing_points=32),
    #     nn.Dropout(dropout),
    #     PMA(256, num_heads=4, num_seeds=16),
    #     nn.Linear(256, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, conditions),
    #     GlobalAvgPool1d(dim=1),
    # )
