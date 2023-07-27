from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .my_colors import ggp2_cyan, ggp2_red, negative_color, positive_color


def draw_hist_feature(data_positive, data_negative, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    min_value = min([data_positive.min(), data_negative.min(), data_positive.max(), data_negative.max()])
    max_value = max([data_positive.min(), data_negative.min(), data_positive.max(), data_negative.max()])
    nbins = max_value - min_value + 1

    log = [False, True]

    if nbins == 2:
        log = None
        bins = [0, 1, 2]

    else:
        bins = np.linspace(min_value, max_value, 100)

    sns.histplot(
        data_negative,
        bins=bins,
        stat="density",
        log_scale=log,
        color=ggp2_cyan,
        alpha=0.7,
        element="step",
    )
    sns.histplot(
        data_positive,
        bins=bins,
        stat="density",
        log_scale=log,
        color=ggp2_red,
        alpha=0.7,
        element="step",
    )

    ax.set_xlim(min_value, max_value)
    ax.tick_params(which="both", direction="in")

    return fig


def draw_model_performance(ytest_pos, ytest_neg, ytrain_pos, ytrain_neg):
    fig, ax = plt.subplots(1, 1)

    sns.histplot(
        ytest_neg,
        label="neg. test",
        bins=100,
        stat="density",
        log_scale=[False, True],
        color=negative_color,
        element="step",
        alpha=0.3,
        linewidth=0,
        ax=ax,
    )
    sns.histplot(
        ytest_pos,
        label="pos. test",
        bins=100,
        stat="density",
        log_scale=[False, True],
        color=positive_color,
        element="step",
        alpha=0.3,
        linewidth=0,
        ax=ax,
    )

    sns.histplot(
        ytrain_pos,
        label="pos. train",
        bins=100,
        stat="density",
        log_scale=[False, True],
        color=positive_color,
        fill=False,
        element="step",
        ax=ax,
    )
    sns.histplot(
        ytrain_neg,
        label="neg. train",
        bins=100,
        stat="density",
        log_scale=[False, True],
        color=negative_color,
        fill=False,
        element="step",
        ax=ax,
    )

    ax.set_xlim(0, 1)
    ax.tick_params(which="both", direction="in")

    ax.legend()

    return fig
