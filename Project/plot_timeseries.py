import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from wordcloud import WordCloud

def word_cloud(cluster, title, font_path='./zh_font.ttf'):
    wc = ' '.join(cluster)
    wordcloud = WordCloud(width=800, height=500,
    random_state=21, max_font_size=110, font_path=font_path).generate(wc)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis('off')
    # plt.savefig(f'zh_cluster_{label + 1}.jpg', pad_inches=0.1, bbox_inches='tight')
    plt.show()

def plot_series(train, cluster, figname, fontpath='./zh_font.ttf', project='zh.wikipedia.org', access='all-access', agent='all-agents'):
    print(f'{figname}\n{cluster}\n')
    dates = train.columns.tolist()[1:]
    f, ax = plt.subplots(figsize=(12, 8))
    font = font_manager.FontProperties(fname=fontpath)

    cluster = cluster.tolist()
    for index, cl in enumerate(cluster):
        if pd.isna(cl):
            break
        data = train[train['Page'] == f'{cl}_{project}_{access}_{agent}']
        array = data.iloc[:, 1:].to_numpy()
        array[pd.isna(array)] = 0

        ax.plot(array.reshape(-1), label=cl)
    
    ax.legend(prop=font)
    ax.set(xlabel='Dates', ylabel='Views')
    ax.set_xticks(np.arange(0, len(dates)))
    ax.set_xticklabels(dates)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))  # set xticks frequency
    # plt.savefig(f'./cluster/correlation/{figname}.jpg')
    plt.show()

    return cluster[:index]

if __name__ == '__main__':
    path = os.getcwd()

    # Read page from train_1
    train = pd.read_csv(f'{path}/data/train_1.csv')
    # print(f'train\n{train}\n')

    # Read cluster data and plot time series
    cluster = pd.read_csv(f'{path}/cluster/zh_cluster_list_100.csv')
    # print(f'cluster\n{cluster}\n')
    # num = len(cluster)
    num = 5
    for i in range(num):
        cl = plot_series(train, cluster.iloc[i], f'cluster{i + 1}')
        word_cloud(cl, f'cluster{i + 1}')