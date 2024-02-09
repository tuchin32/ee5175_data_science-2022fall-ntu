import pandas as pd
import json

def merge_cluster():
    # Read data
    de = pd.read_csv("./cluster/de_pca200_kmeans30.csv", index_col=0)
    en = pd.read_csv("./cluster/en_pca200_kmeans30.csv", index_col=0)
    es = pd.read_csv("./cluster/es_pca200_kmeans30.csv", index_col=0)
    fr = pd.read_csv("./cluster/fr_pca200_kmeans30.csv", index_col=0)
    ja = pd.read_csv("./cluster/ja_pca200_kmeans30.csv", index_col=0)
    ru = pd.read_csv("./cluster/ru_pca200_kmeans30.csv", index_col=0)
    zh = pd.read_csv("./cluster/zh_pca200_kmeans30.csv", index_col=0)
    
    # Merge data
    df = pd.concat([de, en, es, fr, ja, ru, zh], axis=0)
    df = df.reset_index(drop=True)
    
    # Save data as csv file
    df.to_csv("./cluster/cls_pca200_kmeans30.csv", header=False)

    # Save data as csv file
    df_cls = pd.read_csv("./cluster/cls_pca200_kmeans30.csv", index_col=0, header=None)
    print(df_cls.head())
    
    data = {}
    for i in range(df_cls.shape[0]):
        names = df_cls.iloc[i].dropna().to_list()
        cluster = {f"Cluster_{i}": names}
        data.update(cluster)
        
    # Save the data
    with open("./cluster/cls_pca200_kmeans30.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        # json.dump(data, f, indent=4)

def merge_summary():
    # Read data
    de = pd.read_csv("./name/de_summary.csv")
    en = pd.read_csv("./name/en_summary.csv")
    es = pd.read_csv("./name/es_summary.csv")
    fr = pd.read_csv("./name/fr_summary.csv")
    ja = pd.read_csv("./name/ja_summary.csv")
    ru = pd.read_csv("./name/ru_summary.csv")
    zh = pd.read_csv("./name/zh_summary.csv")
    
    # Merge data
    df = pd.concat([de, en, es, fr, ja, ru, zh], axis=0)
    
    # Save data as csv file
    df.to_csv("./name/7language_summary.csv", index=False)

if __name__ == "__main__":
    merge_cluster()
    # merge_summary()
    