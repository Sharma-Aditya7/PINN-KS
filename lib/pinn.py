import tensorflow as tf
from .layer import GradientLayer


class PINN:
    """
    Build a physics informed neural network (PINN) model for the Kuramoto-Sivashinsky equation.

    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        grads: gradient layer.
    """

    def __init__(self, network, *args, **kwargs):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
        """
        super(PINN, self).__init__(*args, **kwargs)
        self.network = network
        self.grads = GradientLayer(self.network, name="GradLayer")

    def build(self):
        """
        Build a PINN model for the KS equation.

        Returns:
            PINN model for the KS equation where:
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition,
                         (t, x=bounds) relative to boundary condition,
                         (t, x) of known points ],
                output: [ u(t,x) relative to equation,
                          u(t=0, x) relative to initial condition,
                          u(t, x=bounds) relative to boundary condition,
                          u(t, x) of known points ],
        """

        # # equation input: (t, x)
        # tx_eqn = tf.keras.layers.Input(shape=(2,), name="Coll_input")
        # # initial condition input: (t=0, x)
        # tx_ini = tf.keras.layers.Input(shape=(2,), name="IC_input")
        # # boundary condition input: (t, x=0)
        # tx_bnd = tf.keras.layers.Input(shape=(2,), name="BC_input")

        tx_all = tf.keras.layers.Input(shape=(8,), name="All_Inputs")
        tx_eqn = tx_all[:, 0:2]
        tx_ini = tx_all[:, 2:4]
        tx_bnd = tx_all[:, 4:6]
        tx_inp = tx_all[:, 6:8]

        # compute gradients
        u, du_dt, du_dx, d2u_dx2, d4u_dx4 = self.grads(tx_eqn)

        # PDE equation (residual)
        u_eqn = du_dt + du_dx*u + d2u_dx2 + d4u_dx4
        # initial condition output
        u_ini = self.network(tx_ini)
        # boundary condition output
        u_bnd = self.network(tx_bnd)  # dirichlet
        # known points output
        u_inp = self.network(tx_inp)

        u_all = tf.concat((u_eqn, u_ini, u_bnd, u_inp), axis=1)
        # u_all = tf.concat((u_eqn, u_ini, u_bnd), axis=1)

        # build the PINN model for the KS equation
        return tf.keras.models.Model(
            inputs=tx_all,
            outputs=u_all)
