# class for tf.keras Fourier features layer
import tensorflow as tf

class ffLayer(tf.keras.layers.Layer):
    """
    Tensorflow keras Layer for Fourier feature extraction

    Attributes
    ----------
    input_dim : int, default 10
        Number of input dimensions of the input.
    m : int, default 100
        Number of desired samples per feature (?).
    sig : array_like, optional
        Array of standard deviations for each matrix.
    B_matrices : array_like
        Array of the randomly sampled values for the feature extraction
        layer. Its shape is `(input_dim, m*len(sig))`.
        Each submatrix of shape `(input_dim, m)` has values taken from
        a zero-mean Gaussian distribution with standard deviation of the
        corresponding `sig`.

    Notes
    ---------
    Only has non-trainable parameters. The matrices are taken by
    considering `m` samples of a zero-mean Gaussian with standard
    deviations taken by `sig`.
    """

    def __init__(self, input_dim=10, m=100, sig=None, B_matrices=None, **kwargs):
        """
        Parameters
        ----------
        input_dim : int, default 10
            Number of input dimensions of the input.
        m : int, default 100
            Number of desired samples per feature (?).
        sig : array_like, optional
            Array of standard deviations for each matrix.
        B_matrices : array_like
            Array of the randomly sampled values for the feature extraction
            layer. Its shape is `(input_dim, m*len(sig))`.
            Each submatrix of shape `(input_dim, m)` has values taken from
            a zero-mean Gaussian distribution with standard deviation of the
            corresponding `sig`.
        """
        super(ffLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.m = m
        if sig is None:
            self.sig = [1, 10, 20, 50, 100]
        else:
            self.sig = sig
        if B_matrices is None:
            self.B_matrices = []
            tf_rng = tf.random.Generator.from_non_deterministic_state()
            for sigi in self.sig:
                self.B_matrices.append(tf_rng.normal(shape=(self.input_dim, self.m),
                                                     mean=0.0, stddev=sigi))
            self.B_matrices = tf.constant(tf.concat(self.B_matrices, 1))
        else:
            self.B_matrices = tf.constant(B_matrices)
        print("", end="")

    def call(self, inputs, *args, **kwargs):
        inp_n_rows = tf.shape(inputs)[0]
        inputs_x_B = tf.matmul(inputs, self.B_matrices)
        # return tf.concat([tf.cos(inputs_x_B), tf.sin(inputs_x_B)], 1)
        aux = tf.concat([tf.cos(inputs_x_B), tf.sin(inputs_x_B)], 0)
        return tf.transpose(tf.reshape(tf.transpose(aux), [self.m*len(self.sig)*2, inp_n_rows]))

    def get_config(self):
        config = super(ffLayer, self).get_config()
        config.update({"input_dim": self.input_dim, "m": self.m,
                       "sig": self.sig, "B_matrices": self.B_matrices.numpy()})
        return config

