# PINN for Kuramoto-Sivashinsky equation
# Breno Vincenzo de Almeida IM458-O
# Adapted from okada39's Github page
# https://github.com/okada39/pinn_burgers

# import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn_weight_anneal import PINN_wa
from lib.network import Network
from lib.fflayer import ffLayer
from matplotlib.colors import LightSource
import time


def plotTimeSeries(data, x, t, cmap='copper', vert_exag=8, fig=None, ax=None):
    # Plot results
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    elif ((fig is None) and not(ax is None)) or (not(fig is None) and (ax is None)):
        raise RuntimeError('Figure and axis handle must both simultaneously given, or None')
    # X, T = np.meshgrid(x, t)
    if cmap == 'redblue':
        ls = LightSource(azdeg=90, altdeg=90)  # Illuminate the scene from the west
        rgb = ls.shade(data, cmap=plt.cm.RdBu, vert_exag=vert_exag, blend_mode='overlay')
        im = ax.imshow(data, cmap=plt.cm.RdBu, origin='lower')
    elif cmap == 'copper':
        ls = LightSource(azdeg=90, altdeg=85)  # Illuminate the scene from the west
        rgb = ls.shade(data, cmap=plt.cm.copper, vert_exag=vert_exag, blend_mode='overlay')
        im = ax.imshow(data, cmap=plt.cm.copper, origin='lower')
    else:
        raise ValueError('Invalid colormap type. Choose between `copper` and `redblue`')
    im.remove()
    fig.colorbar(im, ax=ax)
    ax.imshow(rgb, origin='lower', aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], interpolation=None)
    plt.show(block=False)


if __name__ == '__main__':
    """
    Train a physics informed neural network (PINN) for the Kuramoto-Sivashinsky equation
    """

    # load data
    u_data = np.load("../KSSolution.npy").T
    u_data = np.vstack((u_data, u_data[0:1, :]))
    t_data = np.load("../KSTime.npy")[:, np.newaxis]
    x = np.linspace(0, 200, u_data.shape[0])[:, np.newaxis]

    # divide between training, validation and test sets
    train_test_perc = 0.8
    u_train, u_test = np.split(u_data, [int(t_data.size*train_test_perc)], axis=1)
    t_train, t_test = np.split(t_data, [int(t_data.size*train_test_perc)], axis=0)

    max_t_train = np.max(t_train)

    # number of training samples
    num_train_samples = 20000
    # number of test samples
    num_test_samples = 1000

    rng = np.random.default_rng(seed=1000)

    # # create training output
    # collocation points
    u_coll = np.zeros((num_train_samples, 1))
    # initial conditions
    u_ini = np.tile(u_train[:, 0:1], (int(num_train_samples//u_train.shape[0])+1, 1))[0:num_train_samples, 0:1]
    # boundary conditions
    u_bnd = np.concatenate((u_train[0, 1:], u_train[-1, 1:]))[:, np.newaxis]
    u_bnd = np.tile(u_bnd, (int(num_train_samples//u_bnd.shape[0])+1, 1))[0:num_train_samples, 0:1]
    # data points
    indxs = rng.choice(u_train.size, size=num_train_samples, replace=False)
    u_inp = u_train.ravel()[indxs][:, np.newaxis]
    # # create training input
    # collocation points
    tx_eqn = np.zeros((num_train_samples, 2), dtype=float)
    tx_eqn[:, 0:1] = (max_t_train-t_data[0])*rng.random(size=(num_train_samples, 1)) + t_data[0]
    tx_eqn[:, 1:2] = (x[-1]-x[0])*rng.random(size=(num_train_samples, 1)) + 0.0
    # initial conditions
    tx_ini = np.tile(x, (int(num_train_samples//x.shape[0])+1, 1))[0:num_train_samples, 0:1]
    tx_ini = np.hstack((np.zeros(tx_ini.shape), tx_ini))
    # boundary conditions
    tx_bnd = np.concatenate((np.hstack((t_train[1:], np.zeros(t_train[1:].shape))),
                             np.hstack((t_train[1:], np.full(t_train[1:].shape, 200.)))), axis=0)
    tx_bnd = np.tile(tx_bnd, (int(num_train_samples//tx_bnd.shape[0])+1, 1))[0:num_train_samples]
    # data points
    t_mesh, x_mesh = np.meshgrid(t_train.ravel(), x.ravel())
    tx_inp = np.column_stack((t_mesh.ravel()[indxs], x_mesh.ravel()[indxs]))
    del x_mesh
    del t_mesh
    # build a core network model
    network = Network.build(layers=[50]*4,
                            sig_t=[1, 10],
                            sig_x=[1, 20])
    # network = Network.build(layers=[50]*4)
    network.summary()
    # build a PINN model
    pinn = PINN_wa(network, num_df_terms=3, l_rate=0.001, alpha=0.9)
    # tf.keras.utils.plot_model(pinn, to_file='./model.png', expand_nested=True)

    # train the model using L-BFGS-B algorithm
    x_train = np.concatenate((tx_eqn, tx_ini, tx_bnd, tx_inp), axis=1)
    y_train = np.concatenate((u_coll, u_ini, u_bnd, u_inp), axis=1)

    # train the model using built-in ADAM
    pinn.fit(x_train, y_train, batch_size=5000, epochs=10, verbose=1)

    # train the model using the custom fit, with varying gradients
    history = pinn.custom_fit(x_train, y_train, batch_size=5000, epochs=10)

    # plot loss history and lbda histories
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(history["loss"])
    ax.grid(which='both', color='0.8', linestyle='--')
    ax.set_title("Loss function")
    ax.set_ylabel("Value")
    ax.set_xlabel("Epoch")

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    label_list = ["init. cond.", "bound. cond.", "known points"]
    for i in range(history["lbda"].shape[1]):
        ax.plot(history["lbda"][:, i], label=label_list[i])
    ax.grid(which='both', color='0.8', linestyle='--')
    ax.set_title("Gradient normalization values")
    ax.set_ylabel("Value")
    ax.set_xlabel("Epoch")
    ax.legend()

    # save trained network
    network.save("dense_net_ks")

    # load trained network (testing):
    network = tf.keras.models.load_model("./dense_net_ks", custom_objects={"ffLayer": ffLayer})

    # predict u(t,x) distribution
    t_flat = np.copy(t_data)
    x_flat = np.copy(x)
    T, X = np.meshgrid(t_flat, x_flat)
    tx = np.stack([T.flatten(), X.flatten()], axis=-1)
    U = network.predict(tx, batch_size=1000)
    U = U.reshape(T.shape)

    # plot u(t,x) distribution as a color-map
    #  and plot u(t=const, x) cross-sections
    fig, ax = plt.subplots(1, 3, sharey='all', figsize=(10, 4))
    plotTimeSeries(u_data, X, T, fig=fig, ax=ax[0])
    ax[0].set_title("Original")
    ax[0].set_ylabel("Time")
    ax[0].set_xlabel("x")
    ax[0].set_ylim(500, 2000)
    ax[0].set_yticks(np.insert(np.arange(600, 2200, 200), 0, 500))
    plotTimeSeries(U, X, T, fig=fig, ax=ax[1])
    ax[1].set_title("Predicted")
    ax[1].set_ylabel("Time")
    ax[1].set_xlabel("x")
    rel_err = (np.abs((U - u_data)) / np.abs((u_data + 1.0))).T
    rel_err[rel_err > 1.0] = 1.0
    plotTimeSeries(rel_err, X, T, fig=fig, ax=ax[2])
    ax[2].set_title("Relative difference")
    ax[2].set_ylabel("Time")
    ax[2].set_xlabel("x")

    plt.show(block=True)

    print("Done!")
