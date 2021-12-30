import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

### Constants
from params import \
    N_STEPS as n_steps,\
    IMAGES_PATH

# to make this notebook's output stable across runs
np.random.seed(42)


# # # # # # # # # #
### Time series data
# # # # # # # # # #

def generate_time_series(
        batch_size : int,
        n_steps : int,
):
    """

    "This function creates as many time series as requested (via the batch_size argument),
    each of length n_steps, and there is just one value per time step in each series (i.e.,
    all series are univariate). The function returns a NumPy array of shape
        [batch size, time steps, 1],
    where each series is the sum of two sine waves of fixed amplitudes but random
    frequencies and phases, plus a bit of noise." -- AG p. 504

    NOTE: "When dealing with time series (and other types of sequences such as sentences),
    the input features are generally represented as 3D arrays of shape
        [batch size, time steps, dimensionality],
    where dimensionality is 1 for univariate time series and more for multivariate time series."
        -- AG p. 504

    Parameters
    ----------
    batch_size
        Number distinct time series requested.
    n_steps
        Length of each time series

    Returns
    -------

    """
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)


# # # # # # #
### Plotting
# # # # # # #

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_series(
        series,
        y=None,
        y_pred=None,
        x_label="$t$",
        y_label="$x(t)$",
        legend=True
):
    """
    AG's convenience function to do clean time series plots.
    """
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bo", label="Target")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    if legend and (y or y_pred):
        plt.legend(fontsize=14, loc="upper left")


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "bo-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "rx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)