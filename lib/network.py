import tensorflow as tf
from .fflayer import ffLayer

class Network:
    """
    Build a physics informed neural network (PINN) model for the
    Kuramoto-Sivashinsky equation.
    """

    @classmethod
    def build(cls, num_inputs=2, layers=None, activation=tf.nn.tanh,
              sig_t=[1], sig_x=[1], num_outputs=1, name="DenseNN"):
        """
        Build a PINN model for the Kuramoto-Sivashinsky equation with input
        shape (t, x) and output shape u(t, x).

        Parameters:
            num_inputs: int
                Number of input variables. Default is 2 for (t, x).
            layers: array_like
                List of length equal to number of hidden layers, with the
            number of nodes for each of them.
            activation: stror tensorflow activation object
                Activation function in hidden layers. Default is tanh
            sig_t: array_like of ints
                Standard deviations for the time-domain Fourier feature
                layer.
            sig_x: array_like of ints
                Standard deviations for the spatial-domain Fourier
                feature layer.
            num_outputs: int
                Number of output variables. Default is 1 for u(t, x).
            name : str
                Name of the neural network. Default is "DenseNN"

        Returns:
            keras network model.
        """

        if layers is None:
            layers = [40, 40, 40, 40]

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,), name="t_x")
        # separate time and space
        t = inputs[:, 0:1]
        x = inputs[:, 1:2]
        # Fourier feature layer for time
        t = ffLayer(input_dim=1, m=layers[0], sig=sig_t, name="Time_Fourier_features")(t)
        # Fourier feature layer for space
        x = ffLayer(input_dim=1, m=layers[0], sig=sig_x, name="Space_Fourier_features")(x)
        # dense neural network
        fnn = tf.keras.models.Sequential()
        assert len(sig_t)==len(sig_x)
        fnn.add(tf.keras.layers.Input(shape=(layers[0]*2*len(sig_t))))
        # hidden layers
        for layer in layers:
            fnn.add(tf.keras.layers.Dense(layer, activation=activation,
                                          kernel_initializer='he_uniform',
                                          bias_initializer='he_uniform'))
        # forward pass for time and space
        t = fnn(t)
        x = fnn(x)
        # point-wise multiplication layer for a merge
        tx = tf.multiply(t, x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
                                        kernel_initializer='glorot_uniform')(tx)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)
