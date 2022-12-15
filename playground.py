
import torch
import torch.nn as nn

import pytorch_lightning as lightning


class Base:
    def __init__(self, *args, **kwargs):
        print("Base", args, kwargs)


class Trainable(Base):
    def __init__(self, **hparams):
        print("Trainable", hparams)
        super().__init__()

    def loss(self):
        raise NotImplementedError


class Rectifier(Base):
    def __init__(self, **hparams):
        print("Rectifier", hparams)
        super().__init__(**hparams)

    def loss(self):
        return 0


class FuzzyRectifier(Rectifier, Trainable):
    pass


hparams = dict(
    asdf=3
)

r = FuzzyRectifier(**hparams)

print(r.loss())
