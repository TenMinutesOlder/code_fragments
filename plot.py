import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# df format n*[x_index, y_index, values]
# ind_list = [x_name, y_name]
# val_col = name of value column

def barplot_3d(df, ind_list, val_col):
    df_unstack = df.set_index(ind_list)
    df_unstacked = df_unstack[val_col].unstack()
    cnt_max = df_unstacked.max(axis=1)
    df_unstacked = df_unstacked.apply(lambda x: x / cnt_max, axis=0)
    array = df_unstacked.values

    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax1 = fig.add_subplot(111, projection='3d')

    xlabels = df_unstacked.columns.values
    xpos = np.arange(xlabels.shape[0])
    ylabels = df_unstacked.index.values
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = array
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = (dz - dz.min()) / np.float_(dz.max() - dz.min())
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
    plt.show()


def plot_clusters(ts_array, y_pred, km, sample_size=1000):
    # ts_array is an array in the shape (num_rec, ts_len, 1)
    # y_pred is an 1D array of labels
    # km is a fitted clustering class from tslearn.clustering
    def get_proportion(y_pred):
        df_y = pd.DataFrame(y_pred, columns=['y'])
        df_p = df_y['y'].groupby(df_y['y']).count() / df_y.shape[0] * 100
        return df_p

    sample_list = random.sample(range(ts_array.shape[0]), sample_size)
    ts_sample = ts_array[np.array(sample_list)]
    y_sample = y_pred[np.array(sample_list)]
    df_p = get_proportion(y_pred)
    plt.figure()
    for yi in range(6):
        ax = plt.subplot(2, 3, yi + 1)
        ax.set_title('cluster k=%d \n proportion = %.1f' % (yi, df_p[yi]))
        for xx in ts_sample[y_sample == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2, linewidth=0.9)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-", linewidth=2.5)
        plt.xlim(0, 9)
        plt.ylim(-4, 4)
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    return
