import wikipedia
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def get_wiki_summary(queue, lang):
    print(f'Getting summary for {len(queue)} pages.')

    wikipedia.set_lang(lang)
    summaries = []
    for i, query in enumerate(tqdm(queue)):

        # Get the summary of the page
        # If the page is not found or wikipedia.exceptions.DisambiguationError, return an empty string
        try:
            summaries.append(wikipedia.summary(query, sentences=1))
            if len(summaries[-1]) < 20:
                summaries[-1] = wikipedia.summary(query, sentences=3)
        except wikipedia.exceptions.DisambiguationError:
            summaries.append("DisambiguationError")
        except wikipedia.exceptions.PageError:
            summaries.append("PageError")
        except KeyError:
            summaries.append("KeyError")
        # summaries.append(wikipedia.summary(query, sentences=1))
    
        if (i + 1) % 100 == 0:
            # print(f'{i + 1}/{len(queue)} pages done.')
            # print(summaries[-1])
            output = pd.DataFrame({"name": queue[:(i + 1)], "summary": summaries})
            output.to_csv(f"./name/checkpoints/{lang}_sum_checkpt{i + 1}.csv", index=False)

    print(summaries[:10])
    return summaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', type=str, default='zh')
    # parser.add_argument('--input', '-i', type=str, default='./name/zh_name.csv')
    args = parser.parse_args()
    
    # Read the data. E.g. zh_name.csv(name, type)
    language = args.language
    print(f"Getting summary for {language} pages.")
    df = pd.read_csv(f"./name/{language}_name.csv")

    # Get the summary of each page names
    summaries = get_wiki_summary(df["name"], lang=language)

    # Add the summary to the dataframe
    df["summary"] = summaries   
    df.to_csv(f"./name/{language}_summary.csv", index=False)