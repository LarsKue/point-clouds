import sys
import os

import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

np.set_printoptions(suppress=True)

from functools import partial
from tqdm import tqdm

from statistic_networks import AttentiveStatisticNetwork
from losses import maximum_mean_discrepancy_code



def normalize(points):
    """Normalizes each point cloud (centers around 0 and divides by max norm)."""

    norm_points = points - tf.reduce_mean(points, axis=1, keepdims=True)
    norm_points /= tf.reduce_max(tf.norm(norm_points, axis=-1, keepdims=True), axis=1, keepdims=True)
    return norm_points


def noise(points, mean=0, stddev=0.005):
    """Adds random Gaussian jitter to a shape. Assumes normalized point clouds."""

    points += tf.random.uniform(points.shape, -0.01, 0.01)
    return points


def gen_rotation_matrix(theta):
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]])
    return rot_matrix.astype(np.float32)


def rotate_along_z(points):
    """Rotates each pointcloud along the z-axis."""

    thetas = np.random.rand(points.shape[0]) * 2. * np.pi
    rot_mats = tf.stack([gen_rotation_matrix(theta) for theta in thetas])
    rot_points = tf.matmul(points, rot_mats)
    return rot_points


def transform(points):
    return subsample(noise(normalize(points)))


def subsample(points, n_sub=2048):
    idx = np.random.permutation(n_sub)
    return tf.gather(points, idx, axis=1)


def to_datasets(train_points, test_points, train_labels, test_labels):
    # Create tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    train_dataset = train_dataset.shuffle(len(train_points)).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)
    return train_dataset, test_dataset


def train_epoch(network, optimizer, dataset, batch_size, p_bar, mmd_weight=0.5):
    """ Helper function for one simulation epoch. """

    norms_l = []
    codes_l = []
    for bi, (points, labels) in enumerate(dataset):
        # Encode and decode (ignore latent code for now)
        points = transform(points)

        # Sample time vector
        t = tf.random.uniform((points.shape[0], points.shape[1], 1))

        # Sample x0
        x0 = tf.random.normal(points.shape)

        # Track gradients
        with tf.GradientTape() as tape:
            z, norms = network(x0, points, t, training=True)
            code = maximum_mean_discrepancy_code(z, weight=mmd_weight)
            norm = tf.reduce_mean(norms)
            loss = norm + code

        # Backprop step
        g = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(g, network.trainable_variables))
        norms_l.append(norm.numpy())
        codes_l.append(code.numpy())

        # Update progress bar
        p_bar.set_postfix_str(
            "Step {},L: {:.3f},Avg.L: {:.3f}, MMD: {:.3f},Avg.MMD: {:.3f}".format(bi + 1, norm.numpy(),
                                                                                  np.mean(norms_l), code.numpy(),
                                                                                  np.mean(codes_l)))
        p_bar.update(1)
    return (norms_l, codes_l)


class DriftNetwork(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DriftNetwork, self).__init__(**kwargs)
        self.drift = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='selu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(256, activation='selu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(128, activation='selu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(64, activation='selu', kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(3)
        ])

    def call(self, x0, x1, t, C, **kwargs):
        """Perform step according to: https://arxiv.org/pdf/2209.03003.pdf"""

        N = x0.shape[1]
        C = tf.stack([C] * N, axis=1)
        diff = x1 - x0
        wdiff = t * x1 + (1 - t) * x0
        inp = tf.concat([wdiff, t, C], axis=-1)
        v = self.drift(inp, **kwargs)
        return tf.norm(diff - v, axis=-1)


class SetTransformerFlow(tf.keras.Model):
    def __init__(self):
        super(SetTransformerFlow, self).__init__()

        self.encoder = AttentiveStatisticNetwork(dim=128, name='encoder')
        self.decoder = DriftNetwork(name='drift')

    def call(self, x0, x1, t, **kwargs):
        z = self.encoder(x1, **kwargs)
        norm = self.decoder(x0, x1, t, z, **kwargs)
        return z, norm


if __name__ == '__main__':

    # Train-test split 90 - 10p
    train_points, test_points, train_labels, test_labels = pickle.load(open('data/modelnet10_4096', 'rb+'))

    # Hyperparams train
    network = SetTransformerFlow()
    batch_size = 32
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, 2000, t_mul=1.5, m_mul=0.95)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    epochs = 1000

    # Get data
    train_dataset, test_dataset = to_datasets(train_points, test_points, train_labels, test_labels)

    # Train
    losses = []
    for ep in range(1, epochs + 1):
        with tqdm(total=len(train_dataset), desc=f'Training Epoch {ep}') as p_bar:
            loss_ep = train_epoch(network, optimizer, train_dataset, batch_size, p_bar)
            losses.append(loss_ep)
        network.save_weights('checkpoints/diffusion')
