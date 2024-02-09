import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca(data, n_components=200):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_pca = pca.transform(data)
    return data_pca

def tsne(data, n_components=3):
    tsne = TSNE(n_components=n_components, init='pca', random_state=5, verbose=1)
    tsne.fit(data)
    data_tsne = tsne.fit_transform(data)
    return data_tsne

def kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=5)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    labels = dbscan.labels_
    return labels

def plot_clustering(data, labels, num_clusters):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=5, cmap='rainbow')
    plt.title(f'Semantic-based clustering of Wikipedia pages, number of clusters = {num_clusters}')
    plt.savefig(f'./cls_{num_clusters}.jpg', dpi=300)
    plt.show()
    
def show_cluster(name, labels, num_clusters):
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(labels):
        clustered_sentences[cluster_id].append(name[sentence_id])

    cluster_list_df = pd.DataFrame()
    for i, cluster in enumerate(clustered_sentences):
        print(f'Cluster {i + 1}\n{cluster}\n')
        cluster_list_df = cluster_list_df.append([cluster], ignore_index=True)

    cluster_list_df.to_csv(f'./cluster/{args.language}_pca200_kmeans{num_clusters}.csv')

def word_cloud(pred_df, label, font_path='./zh_font.ttf'):
    wc = ' '.join([text for text in pred_df['name'][pred_df['cluster'] == label]])
    wordcloud = WordCloud(width=500, height=500, colormap='twilight',
                          random_state=21, max_font_size=110, font_path=font_path).generate(wc)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    # plt.title(f'Cluster {label + 1}')
    plt.axis('off')
    # plt.savefig(f'cluster_{args.language}.jpg', dpi=300, pad_inches=0.1, bbox_inches='tight')
    plt.show()
    
def other_trial():
    embeddings = np.load('./name/7language_embedding.npy')
    embeddings = tsne(embeddings, n_components=2)
    names = pd.read_csv('./name/7language_summary.csv')['name']
    
    clusters = pd.read_csv('./cluster/cls_pca200_kmeans30.csv', index_col=0, header=None)
    labels = np.zeros(len(names))
    for i in range(len(clusters)):
        label_idx = np.where(np.isin(names, clusters.iloc[i, :]))[0]
        labels[label_idx] = i
        
    plot_clustering(embeddings, labels, len(clusters))
    
    
    
def main():
    # Read data
    filename = f'./name/{args.language}_summary.csv'
    embedding = f'./name/{args.language}_embedding.npy'
    df = pd.read_csv(filename)
    embeddings = np.load(embedding)
    
    # Feature reduction
    if args.reduction == 'pca':
        embeddings_r = pca(embeddings)
    elif args.reduction == 'tsne':
        embeddings_r = tsne(embeddings)
    else:
        embeddings_r = embeddings
    # embeddings_r = embeddings_r / np.linalg.norm(embeddings_r, axis=1, keepdims=True)
    
    # Clustering
    if args.clustering == 'kmeans':
        cluster_assignment = kmeans(embeddings_r, args.num_clusters)
        num_clusters = args.num_clusters
    elif args.clustering == 'dbscan':
        cluster_assignment = dbscan(embeddings_r)
        num_clusters = len(set(cluster_assignment)) #- (1 if -1 in cluster_assignment else 0)
    else:
        cluster_assignment = np.zeros(len(embeddings_r))
        num_clusters = 1
    
    # Cluster visualization
    df['cluster'] = cluster_assignment
    
    embeddings_r = tsne(embeddings, n_components=2)
    # plot_clustering(embeddings_r, cluster_assignment, num_clusters)
    show_cluster(df['name'], cluster_assignment, num_clusters)
    
    k_list = [0]
    for k in k_list:
        word_cloud(df, k, font_path='./zh_font.ttf')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', type=str, default='zh')
    parser.add_argument('--reduction', '-r', type=str, default='pca')
    parser.add_argument('--clustering', '-c', type=str, default='kmeans')
    parser.add_argument('--num_clusters', '-k', type=int, default=30)
    # parser.add_argument('--filename', '-f', type=str, default='./name/zh_summary.csv')
    # parser.add_argument('--embedding', '-e', type=str, default='./name/zh_embedding.npy')
    args = parser.parse_args()
    
    # main()
    other_trial()
    
    