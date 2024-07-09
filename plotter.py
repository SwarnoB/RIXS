import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(df1, df2, labels=['unpumped', 'pumped'], title='XES of liquid water', xlabel='eV', ylabel=''):
    fig, ax = plt.subplots()
    ax.plot(df1.En, df1.inc, label=f'{labels[0]} (inc)')
    ax.plot(df1.En, df1.coh, label=f'{labels[0]} (coh)', linestyle='--')
    ax.plot(df2.En, df2.inc, label=f'{labels[1]} (inc)')
    ax.plot(df2.En, df2.coh, label=f'{labels[1]} (coh)', linestyle='--')
    ax.legend()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def plot_norm_spec_coh(df1, df2, labels=['unpumped water', 'pumped water'], title='XES of liquid water', xlabel='eV', ylabel=''):
    fig, ax = plt.subplots()
    ax.plot(df1.En, df1.coh, label=f'{labels[0]} (coh)', linestyle='--')
    ax.plot(df2.En, df2.coh, label=f'{labels[1]} (coh)', linestyle='--')
    ax.legend()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def plot_xy(ax, x, y1, y2):
    """
    Plots a single set of x, y data on a given Axes object.

    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - x: X-axis data.
    - y: Y-axis data.
    - label: Label for the dataset.
    """
    ax.plot(x, y1, label='unpumped water', color='black')
    ax.plot(x, y2, label='pumped water', color='red')
    ax.legend(frameon=False)
    #ax.grid(True)

def plot_three_sets(x, y11, y12, y21, y22, y31, y32, title=None, xlabel="X-axis", ylabel="Y-axis"):
    """
    Sets up the figure and plots three sets of x, y data in a grid of subplots.

    Parameters:
    - x1, y1: First set of x, y data.
    - x2, y2: Second set of x, y data.
    - x3, y3: Third set of x, y data.
    - labels: List of labels for the datasets, e.g., ["Set 1", "Set 2", "Set 3"].
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    fig, axs = plt.subplots(3, 1, figsize=(5, 7.5), gridspec_kw = {'hspace':0})

    plot_xy(axs[0], x, y11, y12)
    plot_xy(axs[1], x, y21, y22)
    plot_xy(axs[2], x, y31, y32)

    for indx, ax in enumerate(axs):
        ax.set_xlim(515, 540)
        #ax.set_ylim(0, 60000)
        ax.set_ylim(0.,)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        #ax.text(517, 50000, title[indx])
        if indx==0:
            ax.text(517,0.25, title[indx])
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        elif indx==1:
            ax.text(517, 0.50, title[indx])
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        else:
            ax.set_ylim(0, 6.5)
            ax.text(517, 5.0, title[indx])
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    
    for indx, ax in enumerate(axs):
        ax.label_outer()
    for ax in axs[:-1]:
        ax.xaxis.set_tick_params(labelbottom=False)

    #for ax in axs:
    #    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    #    ax.tick_params(axis='y', which='both', left=True, right=False)

    #axs[-1].tick_params(axis='x', which='both', labelbottom=True)
    #plt.subplots_adjust(hspace=5.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    #plt.savefig('water_RIXS_unnormalized.png', transparent=True, dpi=300)
