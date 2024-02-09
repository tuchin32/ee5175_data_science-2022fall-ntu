import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def correlation(train, cluster, figname, location=(0, 1), project='zh.wikipedia.org', access='all-access', agent='all-agents'):
    # Choose two time series
    print(f'{figname}\n{cluster}\n')
    cluster1 = cluster[location[0]]
    cluster2 = cluster[location[1]]
    data1 = train[train['Page'] == f'{cluster1}_{project}_{access}_{agent}']
    data2 = train[train['Page'] == f'{cluster2}_{project}_{access}_{agent}']
    dates = train.columns.tolist()[1:]

    # Compute correlation
    array1 = data1.iloc[:, 1:].to_numpy()
    array2 = data2.iloc[:, 1:].to_numpy()
    array1[pd.isna(array1)] = 0
    array2[pd.isna(array2)] = 0
    
    df = pd.DataFrame()
    df[cluster1] = array1.reshape(-1)
    df[cluster2] = array2.reshape(-1)
    correlation = df.corr().iloc[0,1]
    
    # Part 1: Compute the overall correlation
    font = font_manager.FontProperties(fname="./zh_font.ttf")
    f, ax = plt.subplots(figsize=(12, 8))
    df.rolling(window=7, center=True).median().plot(ax=ax)
    ax.plot(df[cluster1], color='C0', alpha=0.2, label=cluster1+'(time series)')
    ax.plot(df[cluster2], color='C1', alpha=0.2, label=cluster2+'(time series)')
    ax.legend(prop=font)
    ax.set(xlabel='Dates', ylabel='Median views (7-day rolling window)')
    ax.set(title=f"Overall correlation coefficient r = {np.round(correlation, 3)}")
    ax.set_xticks(np.arange(0, len(dates)))
    ax.set_xticklabels(dates)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))  # set xticks frequency
    plt.savefig(f'./cluster/correlation/{figname}.jpg')
    plt.show()
    # df.to_csv(f'./cluster/ex1.csv', index=False, header=True)


    # Part 2: Compute correlation coefficient with rolling window
    rolling_r = df[cluster1].rolling(window=15, center=True).corr(df[cluster2])
    f, ax=plt.subplots(2, 1, figsize=(12, 8),sharex=True)
    df.rolling(window=7, center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Dates', ylabel='Median views (7-day rolling window)')
    ax[0].legend(prop=font)
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Dates', ylabel='Correlation (15-day rolling window)')
    ax[1].set_xticks(np.arange(0, len(dates)))
    ax[1].set_xticklabels(dates)
    ax[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))   # set xticks frequency
    plt.suptitle("Views and rolling window correlation")
    plt.savefig(f'./cluster/correlation/{figname}_rolling.jpg')
    plt.show()



if __name__ == '__main__':
    path = os.getcwd()

    # Read page from train_1
    train = pd.read_csv(f'{path}/data/train_1.csv')
    print(f'train\n{train}\n')

    # Read cluster data
    cluster = pd.read_csv(f'{path}/cluster/zh_cluster_list_100.csv')
    print(f'cluster\n{cluster}\n')

    # Find the correlation between cluster and page
    correlation(train, cluster.iloc[64], 'zh100_cluster64')
    correlation(train, cluster.iloc[19], 'zh100_cluster19', location=(1, 9))
    correlation(train, cluster.iloc[89], 'zh100_cluster89')

