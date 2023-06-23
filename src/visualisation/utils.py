from src.locations import FIGURES_HOME

import os


def save_plot(plot, name):
    """
    Persist PNG plot in the figures folder

    Parameters:
        plot: a matplotlib plot object
        name: extension-less name for the plot
    """
    plot.savefig(os.path.join(FIGURES_HOME, name + ".png"))
