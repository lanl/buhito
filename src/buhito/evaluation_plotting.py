"""Charts for evaluating arbitrary regression models."""
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error as MAE, mean_squared_error as MSE


def RMSE(true, pred):
    return np.sqrt(MSE(true, pred))


def add_identity(axes, **line_kwargs):
    """https://stackoverflow.com/questions/22104256/"""
    identity, = axes.plot([], [], **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def add_best_fit(x, y, ax, **kwargs):
    """Add a line of best fit to a scatterplot of x and y on ax"""

    # determine best fit line
    poly = np.polyfit(x, y, 1, full=True)

    slope = poly[0][0]
    intercept = poly[0][1]
    xl = [min(x), max(x)]
    yl = [slope*xx + intercept  for xx in xl]
    ax.plot(xl, yl, **kwargs)
    return ax


def scatter_hist(x,
                 y,
                 bins=50,
                 xlab=None,
                 ylab=None,
                 title=None,
                 ax=None,
                 fontsize=None,
                 identity=False,
                 best_fit=True,
                 identity_kws=None,
                 best_fit_kws=None,
                 **kwargs):
    """Create a scatter plot (with identity) and a corresponding 2d histogram

    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    h = ax.hist2d(x, y, bins=bins, norm=matplotlib.colors.LogNorm(), cmap=matplotlib.colormaps['viridis'])
    plt.colorbar(h[3], ax=ax)
    if identity:
        if identity_kws is None:
            identity_kws = dict(color='black', linestyle='-', label='Identity')
        add_identity(ax, **identity_kws)
    if best_fit:
        if best_fit_kws is None:
            best_fit_kws = dict(color='black', linestyle='--', label='Best fit')
        add_best_fit(x, y, ax, **best_fit_kws)
    if title:
        ax.set_title(title)
    return ax

def add_textbox(d,
                ax,
                x=.05,
                y=.95,
                sigfigs=4,
                notation='standard',
                fontsize=14,
                text_alpha=0.9
               ):
    bbox_params = dict(boxstyle='round',
                       facecolor='white',
                       alpha=text_alpha)
    textparams = dict(transform=ax.transAxes,
                      fontsize=fontsize,
                      verticalalignment='top',
                      bbox=bbox_params)
    text = []
    for k, v in d.items():
        v = float(v)
        if math.isnan(v):
            v = 'NaN'
        else:
            import sigfig
            v = sigfig.round(v, sigfigs=sigfigs, notation=notation)

        text.append(f'{k}={v}')

    ax.text(x, y,
            '\n'.join(text),
            **textparams)

def parity_plot(true,
              pred,
              ax=None,
              sigfigs=4,
              notation='standard',
              metrics=None,
              text_loc=(0.05, 0.95),
              legend_loc='lower right',
              text_fontsize=14,
              text_alpha=0.9,
              reverse_met_args=False,
              legend=True,
              identity_kws=None,
              **kwargs):
    """Plot true against pred, with metrics, parity, and a line of best fit

    identity_kws get passed to add_identity
    kwargs are passed to scatter_hist
    """

    allowed_metrics = {'mae': {'display_name': 'MAE',
                               'fn': MAE},
                       'r2': {'display_name': '$R^2$',
                              'fn': r2_score},
                       'rmse': {'display_name': 'RMSE',
                                'fn': RMSE},
                       'pearsonr': {'display_name': 'Pearson $r$',
                                    'fn': lambda x, y: pearsonr(x, y)[0]},
                       'spearmanr': {'display_name': r'Spearman $\rho$',
                                     'fn': lambda x, y: spearmanr(x, y)[0]}
                      }

    if metrics == None:
        metrics = 'mae', 'r2', 'rmse'


    metric_values = {}
    for metric in metrics:
        if metric in allowed_metrics.keys():
            met_args = (true, pred) if not reverse_met_args else (pred, true)
            metric_values[allowed_metrics[metric]['display_name']] = allowed_metrics[metric]['fn'](*met_args)
        else:
            warnings.warn(f'"{metric}" not in allowed metrics ({",".join(list(allowed_metrics.keys()))})')

    ax = scatter_hist(true, pred, ax=ax, identity_kws=identity_kws, **kwargs)
    if metric_values:
        x, y = text_loc
        add_textbox(metric_values, ax, x=x, y=y, sigfigs=sigfigs, notation=notation, fontsize=text_fontsize, text_alpha=text_alpha)

    if legend:
        ax.legend(loc=legend_loc)

    return ax

def parity_plot_train_test(X, y, model,
                         title=None,
                         figsize=(8,3),
                         dpi=None,
                         **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for i, (subset, x, y) in enumerate(zip(['train', 'test'], X, y)):
        pred = model.predict(x)
        parity_plot(y, pred, ax=axes[i], title=f'{subset.title()} (n={y.shape[0]:,d})', **kwargs)
        if title:
            plt.suptitle(title)
    return axes
