import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def read_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    name = df['name'].to_list()
    summary = df['summary'].to_list()
    summary = [str(sum) for sum in summary]
    
    for i in range(len(summary)):
        if summary[i] == 'DisambiguationError':
            summary[i] = name[i]
        elif summary[i] == 'PageError':
            summary[i] = name[i]
        elif summary[i] == 'KeyError':
            summary[i] = name[i]
    
    return name, summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', type=str, default='zh')
    args = parser.parse_args()
    
    # Read the data
    file_path = f"./name/{args.language}_summary.csv"
    name, summary = read_data(file_path)
    
    # Get the embedding of the summary
    embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    embeddings = embedder.encode(summary)
    print('embedding', embeddings.shape)
    print(embeddings)
    
    # Save the embedding as npy file
    np.save(f'./name/{args.language}_embedding.npy', embeddings)
    