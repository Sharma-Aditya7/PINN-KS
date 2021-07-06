import tensorflow as tf


class GradientLayer(tf.keras.layers.Layer):
    """
    Computing the 1st time derivative and 1st and 2nd and 3rd space
    derivatives for the Kuramoto-Sivashinsky equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    @tf.function
    def call(self, tx):
        """
        Computing the 1st time derivative and 1st and 2nd and 3rd space
        derivatives for the Kuramoto-Sivashinsky equation.

        Args:
            tx: input variables (t, x).

        Returns:
            u: network output.
            du_dt: 1st t derivative of u.
            du_dx: 1st x derivative of u.
            d2u_dx2: 2nd x derivative of u.
            d4u_dx4: 4nd x derivative of u.
        """

        t = tx[:, 0:1]
        x = tx[:, 1:2]
        with tf.GradientTape() as gggg:
            gggg.watch(x)
            gggg.watch(t)
            with tf.GradientTape() as ggg:
                ggg.watch(x)
                ggg.watch(t)
                with tf.GradientTape() as gg:
                    gg.watch(x)
                    gg.watch(t)
                    with tf.GradientTape() as g:
                        txaux = tf.concat((t, x), axis=1)
                        g.watch(txaux)
                        u = self.model(txaux)
                    du_dtx = g.batch_jacobian(u, txaux)
                    du_dt = du_dtx[..., 0]
                    du_dx = du_dtx[..., 1]
                # d2u_dtx2 = gg.batch_jacobian(du_dtx, tx)
                # d2u_dx2 = d2u_dtx2[..., 1, 1]
                d2u_dx2 = gg.batch_jacobian(du_dx, x)
            # d3u_dtx3 = ggg.batch_jacobian(d2u_dtx2, tx)
            d3u_dx3 = ggg.batch_jacobian(d2u_dx2, x)
        # d4u_dtx4 = gggg.batch_jacobian(d3u_dtx3, tx)
        # d4u_dx4 = d4u_dtx4[..., 1, 1, 1, 1]
        d4u_dx4 = gggg.batch_jacobian(d3u_dx3, x)

        return u, du_dt, du_dx, d2u_dx2[..., 0], d4u_dx4[..., 0, 0, 0]
