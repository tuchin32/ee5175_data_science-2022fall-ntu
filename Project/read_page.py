import os
import numpy as np
import pandas as pd

def page_processing(path, filename):
    # Read page
    page_df = pd.read_csv(f'{path}/data/{filename}.csv', usecols=['Page'])
    print(page_df)


    # Split page into name, project, access, agent
    pages = page_df['Page'].to_list()

    train = {'name': [], 'project': [], 'access': [], 'agent': []}
    for page in pages:
        page = page.split('_')
        train['project'].append(page[-3])
        train['access'].append(page[-2])
        train['agent'].append(page[-1])
        train['name'].append(' '.join(page[:-3]))

    train_df = pd.DataFrame(data=train)
    train_df = pd.concat([page_df, train_df], axis=1)


    # Save to file
    savename = filename + '_page.csv'
    # train_df.to_csv(os.path.join(path, 'page', savename), index=False, header=True)

    return train_df

def date_processing(path, data_df, filename):
    # Read date
    train_df = pd.read_csv(f'{path}/data/{filename}.csv')
    dates = train_df.to_numpy()
    dates = dates[:, 1:]

    print(dates.shape)
    print(dates)

    # Count total number of views
    dates_zero = dates.copy()
    dates_zero[pd.isna(dates_zero)] = 0
    views = np.sum(dates_zero, axis=1, dtype=np.int64)
    print('views', views)

    # Count number of days with views
    valid_date = dates.shape[1] - np.sum(pd.isna(dates), axis=1)
    print('valid date', valid_date)

    # Merge to data_df and save to file
    data_df['views'] = views
    data_df['valid_date'] = valid_date
    # data_df.to_csv(os.path.join(path, 'page', filename + '_dates.csv'), index=False, header=True)

    return data_df
    


if __name__ == '__main__':
    path = os.getcwd()

    # Read page from train data
    train_1_df = page_processing(path, 'train_1')
    # train_2_df = page_processing(path, 'train_2')
    # key_1_df = page_processing(path, 'key_1')
    # key_2_df = page_processing(path, 'key_2')
    
    train_1_df_new = date_processing(path, train_1_df, 'train_1')
    