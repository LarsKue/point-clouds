import tensorflow as tf


MMD_BANDWIDTH_LIST = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6
]


def _gaussian_kernel_matrix(x, y, sigmas=None):
    """ Computes a Gaussian Radial Basis Kernel between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x      :  tf.Tensor of shape (N, num_features)
    y      :  tf.Tensor of shape (M, num_features)
    sigmas :  list(float), optional, default: None (use default)
        List which denotes the widths of each of the Gaussians in the kernel.

    Returns
    -------
    kernel : tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    norm = lambda v: tf.reduce_sum(tf.square(v), 1)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return kernel


def _inverse_multiquadratic_kernel_matrix(x, y, sigmas=None):
    """ Computes an inverse multiquadratic RBF between the samples of x and y.
    We create a sum of multiple IM-RBF kernels each having a width :math:`\sigma_i`.
    Parameters
    ----------
    x :  tf.Tensor of shape (M, num_features)
    y :  tf.Tensor of shape (N, num_features)
    sigmas : list(float)
        List which denotes the widths of each of the gaussians in the kernel.
    Returns
    -------
    kernel: tf.Tensor
        RBF kernel of shape [num_samples{x}, num_samples{y}]
    """
    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    dist = tf.expand_dims(tf.reduce_sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = tf.expand_dims(sigmas, 0)
    return tf.reduce_sum(sigmas / (dist + sigmas), axis=-1)


def mmd_kernel(x, y, kernel):
    """ Computes the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from 
    the distributions of x and y.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        Random dtaws 
    y      : tf.Tensor of shape (M, num_features)
    kernel : callable 
        A function which computes the kernel for MMD.

    Returns
    -------
    loss   : tf.Tensor
        squared maximum mean discrepancy loss, shape (,)
    """

    loss = tf.reduce_mean(kernel(x, x))  
    loss += tf.reduce_mean(kernel(y, y))  
    loss -= 2 * tf.reduce_mean(kernel(x, y))  
    return loss


def maximum_mean_discrepancy_code(code, n_z_draws=300, latent_dist=tf.random.normal, weight=1., kernel=_gaussian_kernel_matrix):
    """ Computes the MMD between latent code and latent distro.

    Parameters
    ----------
    code     : tf.Tensor or np.ndarray
        Original data of shape (batch_size, latent_dim)

    Returns
    -------
    loss_value : tf.Tensor
        A scalar Maximum Mean Discrepancy, shape (,)
    """
    
    if weight == 0:
        return 0.
    z = latent_dist(shape=(n_z_draws, code.shape[1]))
    mmd = weight * mmd_kernel(code, z, kernel=kernel) 
    return mmd