import tensorflow as tf
import time
from .layer import GradientLayer
import warnings
import numpy as np


class PINN_wa(tf.keras.models.Model):
    """
    Physics informed neural network (PINN) model for the Kuramoto-Sivashinsky
    equation with an adaptive weighted annealing learning rate.

    Attributes
    ----------
    network: tensorflow model layer object
        Tensorflow keras network model with input (t, x) and output u(t, x).
    grads: tesnsorflow keras layer object
        Gradient layer.
    lbda: array_like
        Weights for training.
    num_df_terms: int
        Number of data-fit terms, such as measurements, boundary and
        initial conditions, etc. Can be 2 or 3 only.
    l_rate : float or tensorflow scheduler. Default: 1e-3
        Learning rate for the optimizer. Can be a tensorflow scheduler
    alpha : tensorflow constant. Default: 0.9
        Update momentum for `lbda`.
    """

    def __init__(self, network: tf.keras.models.Model, num_df_terms: int,
                 l_rate=1e-3, alpha=0.9, *args, **kwargs):
        """
        Parameters
        ----------
        network: tensorflow model layer object
            Tensorlfow keras network model with input (t, x) and output u(t, x).
        num_df_terms: int
            Number of data-fit terms, such as measurements, boundary and
            initial conditions, etc. Can be 2 or 3 only.
        l_rate : float or tensorflow scheduler. Default: 1e-3
            Learning rate for the optimizer. Can be a tensorflow scheduler
        alpha : float. Default: 0.9
            Update momentum for `lbda`.
        """
        super(PINN_wa, self).__init__(*args, **kwargs)
        # the non-PINN model, to be trained
        self.network = network
        # the gradient layer
        self.gradLayer = GradientLayer(self.network, name="GradLayer")
        # number of data fit terms (e.g. initial conditions, boundary conditions,
        # known/measured data points, ...)
        self.num_df_terms = num_df_terms
        # aliasing the call function based on self.num_df_terms
        self._calls_dict = {2: self._call2, 3: self._call3}
        self.call = self._calls_dict.get(self.num_df_terms, None)
        if self.call is None:
            raise NotImplementedError("Number of data fit terms should be 2 or 3, not {}".format(self.num_df_terms))
        # gradient scaling variables
        self.lbda = tf.Variable(tf.ones(shape=(self.num_df_terms,)), trainable=False)
        # auxiliary tf.Variable's
        self.aux_vec = tf.Variable(tf.zeros(shape=(self.num_df_terms+1,)), trainable=False)
        # learning rate
        self.l_rate = l_rate
        # compiling the model
        self.compile(loss='mse',
                     optimizer=tf.keras.optimizers.Adam(learning_rate=self.l_rate))
        # aliasing the gradient normalization function based on self.num_df_terms
        self._update_lbda_dict = {2: self._update_lbda2, 3: self._update_lbda3}
        self.update_lbda = self._update_lbda_dict.get(self.num_df_terms, None)
        if self.call is None:
            raise NotImplementedError("Number of data fit terms should be 2 or 3, not {}".format(self.num_df_terms))
        # save alpha as a tensorflow constant
        self.alpha = tf.constant(alpha)

    # call method for two data fit terms
    def _call2(self, tx_all, **kwargs):
        tx_eqn = tx_all[:, 0:2]
        tx_ini = tx_all[:, 2:4]
        tx_bnd = tx_all[:, 4:6]

        # compute gradients
        u, du_dt, du_dx, d2u_dx2, d4u_dx4 = self.gradLayer(tx_eqn)

        # PDE equation (residual)
        u_eqn = du_dt + du_dx*u + d2u_dx2 + d4u_dx4

        # initial condition output
        u_ini = self.network(tx_ini)*self.lbda[0]
        # boundary condition output
        u_bnd = self.network(tx_bnd)*self.lbda[1]
        return tf.concat((u_eqn, u_ini, u_bnd), axis=1)

    # call method for three data fit terms
    def _call3(self, tx_all, **kwargs):
        tx_eqn = tx_all[:, 0:2]
        tx_ini = tx_all[:, 2:4]
        tx_bnd = tx_all[:, 4:6]
        tx_pts = tx_all[:, 6:8]

        # compute gradients
        u, du_dt, du_dx, d2u_dx2, d4u_dx4 = self.gradLayer(tx_eqn)

        # PDE equation (residual)
        u_eqn = du_dt + du_dx*u + d2u_dx2 + d4u_dx4

        # initial condition output
        u_ini = self.network(tx_ini)*self.lbda[0]
        # boundary condition output
        u_bnd = self.network(tx_bnd)*self.lbda[1]
        # measured points
        u_pts = self.network(tx_pts)*self.lbda[2]
        return tf.concat((u_eqn, u_ini, u_bnd, u_pts), axis=1)

    @tf.function
    def custom_train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predicted = self(inputs, training=True)  # forward pass
            loss_val = self.compiled_loss(predicted, targets)
        grads = tape.gradient(loss_val, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_val

    @tf.function
    def _update_lbda2(self, tx_all, u_all):
        tx_eqn = tx_all[:, 0:2]
        tx_ini = tx_all[:, 2:4]
        tx_bnd = tx_all[:, 4:6]

        u_eqn = u_all[:, 0:1]
        u_ini = u_all[:, 1:2]
        u_bnd = u_all[:, 2:3]

        ## get gradient from PDE residual
        with tf.GradientTape() as tape:
            # compute gradients
            u, du_dt, du_dx, d2u_dx2, d4u_dx4 = self.gradLayer(tx_eqn)
            # PDE equation (residual)
            u_eqn_p = du_dt + du_dx*u + d2u_dx2 + d4u_dx4
            loss = self.compiled_loss(u_eqn_p, u_eqn)
        grad_eqn = tape.gradient(loss, self.network.trainable_variables)  # this returns a list of gradients
        self.aux_vec[0].assign(0.)
        for grad in grad_eqn:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[0], true_fn=lambda: self.aux_vec[0].assign(aux), false_fn=lambda: 0.)

        ## get gradient from data-fit values
        with tf.GradientTape() as tape:
            # predicted initial conditions
            u_ini_p = self.network(tx_ini)
            loss = self.compiled_loss(u_ini_p, u_ini)
        grad_ini = tape.gradient(loss, self.network.trainable_variables)
        self.aux_vec[1].assign(0.)
        for grad in grad_ini:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[1], true_fn=lambda: self.aux_vec[1].assign(aux), false_fn=lambda: 0.)
        with tf.GradientTape() as tape:
            # predicted boundary conditions
            u_bnd_p = self.network(tx_bnd)
            loss = self.compiled_loss(u_bnd_p, u_bnd)
        grad_bnd = tape.gradient(loss, self.network.trainable_variables)
        self.aux_vec[2].assign(0.)
        for grad in grad_bnd:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[2], true_fn=lambda: self.aux_vec[2].assign(aux), false_fn=lambda: 0.)

        # update `self.lbda`
        tf_one = tf.constant(1.0)
        self.lbda[0].assign((tf_one-self.alpha)*self.lbda[0] + self.alpha*self.aux_vec[0]/self.aux_vec[1])
        self.lbda[1].assign((tf_one-self.alpha)*self.lbda[1] + self.alpha*self.aux_vec[0]/self.aux_vec[2])

    @tf.function
    def _update_lbda3(self, tx_all, u_all):
        tx_eqn = tx_all[:, 0:2]
        tx_ini = tx_all[:, 2:4]
        tx_bnd = tx_all[:, 4:6]
        tx_pts = tx_all[:, 6:8]

        u_eqn = u_all[:, 0:1]
        u_ini = u_all[:, 1:2]
        u_bnd = u_all[:, 2:3]
        u_pts = u_all[:, 3:4]

        ## get gradient from PDE residual
        with tf.GradientTape() as tape:
            # compute gradients
            u, du_dt, du_dx, d2u_dx2, d4u_dx4 = self.gradLayer(tx_eqn)
            # PDE equation (residual)
            u_eqn_p = du_dt + du_dx*u + d2u_dx2 + d4u_dx4
            loss = self.compiled_loss(u_eqn_p, u_eqn)
        grad_eqn = tape.gradient(loss, self.network.trainable_variables)  # this returns a list of gradients
        self.aux_vec[0].assign(0.)
        for grad in grad_eqn:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[0], true_fn=lambda: self.aux_vec[0].assign(aux), false_fn=lambda: 0.)

        ## get gradient from data-fit values
        with tf.GradientTape() as tape:
            # predicted initial conditions
            u_ini_p = self.network(tx_ini)
            loss = self.compiled_loss(u_ini_p, u_ini)
        grad_ini = tape.gradient(loss, self.network.trainable_variables)
        self.aux_vec[1].assign(0.)
        for grad in grad_ini:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[1], true_fn=lambda: self.aux_vec[1].assign(aux), false_fn=lambda: 0.)
        with tf.GradientTape() as tape:
            # predicted boundary conditions
            u_bnd_p = self.network(tx_bnd)
            loss = self.compiled_loss(u_bnd_p, u_bnd)
        grad_bnd = tape.gradient(loss, self.network.trainable_variables)
        self.aux_vec[2].assign(0.)
        for grad in grad_bnd:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[2], true_fn=lambda: self.aux_vec[2].assign(aux), false_fn=lambda: 0.)
        with tf.GradientTape() as tape:
            # predicted measured points
            u_pts_p = self.network(tx_pts)
            loss = self.compiled_loss(u_pts_p, u_pts)
        grad_pts = tape.gradient(loss, self.network.trainable_variables)
        self.aux_vec[3].assign(0.)
        for grad in grad_pts:
            aux = tf.reduce_max(tf.abs(grad))
            tf.cond(aux > self.aux_vec[3], true_fn=lambda: self.aux_vec[3].assign(aux), false_fn=lambda: 0.)

        # update `self.lbda`
        tf_one = tf.constant(1.0)
        self.lbda[0].assign((tf_one-self.alpha)*self.lbda[0] + self.alpha*self.aux_vec[0]/self.aux_vec[1])
        self.lbda[1].assign((tf_one-self.alpha)*self.lbda[1] + self.alpha*self.aux_vec[0]/self.aux_vec[2])
        self.lbda[2].assign((tf_one-self.alpha)*self.lbda[2] + self.alpha*self.aux_vec[0]/self.aux_vec[3])

    def custom_fit(self, x, y, batch_size=32, epochs=1):
        # guarantee that `epochs` is an int
        if type(epochs) is not int:
            warnings.warn("WARNING: epochs should be an int. Casting it to int.", RuntimeWarning)
            epochs = int(epochs)

        # list holding the loss and lbda histories
        loss_hist = np.zeros(epochs, dtype=float)
        lbda_hist = np.zeros((epochs, self.num_df_terms), dtype=float)

        # prepare training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        x = tf.constant(x)
        y = tf.constant(y)
        train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True)  # not sure if this kwarg works

        # start measuring the time to train the network
        start_time = time.perf_counter()

        # begin training the network
        for epoch in range(epochs):
            print("Epoch {:d}/{:d}:".format(epoch+1, epochs))

            # shuffle the dataset at the start of each epoch
            train_dataset_epoch = train_dataset.batch(batch_size)  # not sure if this will shuffle everything each epoch
            steps = len(train_dataset_epoch)

            # calculate the `lbda` values
            # I have to use self.lbda[0].assign([value]) to assign the values to the tf.Variable `self.lbda`
            print("Updating the gradient normalization values...", end="")
            start_time_lbda = time.perf_counter()
            self.update_lbda(x, y)
            time_lbda_duration = time.perf_counter() - start_time_lbda
            print(" {:.3f} s".format(time_lbda_duration))

            # # shuffle data at the start of each epoch
            # train_dataset_epoch = train_dataset.shuffle(1024, reshuffle_each_iteration=True).batch(batch_size)
            # steps = len(train_dataset)
            print("{:d}/{:d} [{:30}] - {:.0f}s {:.0f}ms/step - loss: {:.4e}".format(0, steps, "="*int(1*30//steps),
                                                                                    0.0, 0.0, 0.0), end="")
            acc_loss = 0.0
            start_time_epoch = time.perf_counter()
            # optimize in the epoch, per batch
            for step, (x_batch, y_batch) in enumerate(train_dataset_epoch):
                start_time_step = time.perf_counter()
                # perform the optimization here
                acc_loss += self.custom_train_step(x_batch, y_batch)
                end_time_step = time.perf_counter()
                step_duration = end_time_step-start_time_step
                epoch_duration = end_time_step - start_time_epoch
                print("\r{:d}/{:d} [{:30}] - {:.0f}s {:.0f}ms/step - loss: {:.4e}".format(step+1, steps, "="*int((step+1)*30//steps),
                                                                                          epoch_duration, step_duration*1000,
                                                                                          acc_loss/(step+1)), end="")
            print("")

            # save the loss and lbda values of this epoch
            loss_hist[epoch] = acc_loss/steps
            lbda_hist[epoch, :] = self.lbda.numpy()

        # print training duration
        end_time = time.perf_counter()
        training_duration = end_time - start_time
        training_duration = {"h": int(training_duration // 3600),
                             "min": int((training_duration % 3600) // 60),
                             "sec": (training_duration % 3600) % 60}
        print("Training duration was {h} h, {min} min, {sec:.3f} sec.".format(**training_duration))

        history = {"loss": loss_hist,
                   "lbda": lbda_hist}
        return history
