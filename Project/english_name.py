import os
import googletrans
import numpy as np
import pandas as pd

if __name__ == '__main__':
    path = os.getcwd()

    # Read page from train_1
    train = pd.read_csv(f'{path}/page/train_1_page.csv')
    print(f'{train}\n')

    names = train['name'].unique()
    print(f'Name has {len(names)} types\n')

    # Translate name to English
    translator = googletrans.Translator()
    # train['name_en'] = train['name'].apply(translator.translate, dest='en').apply(getattr, args=('text',))
    # print(translator.translate('我覺得今天天氣不好', dest='en').text)

    name_en = []
    names_sub = names[21000:23000]
    for i, name in enumerate(names_sub.tolist()):
        if i % 500 == 0:
            print(i)

        detect = translator.detect(name)
        if detect.lang == 'en':
            name_en.append(name)
        else:
            trans = translator.translate(name, dest='en').text
            # print(trans)
            name_en.append(trans)

    name_df = pd.DataFrame(data={'name': names_sub, 'name_en': name_en})

    # Save to file
    name_df.to_csv(os.path.join(path, 'page', 'train_1_name_en_1.csv'), index=False, header=True)