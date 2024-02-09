import os
import numpy as np
import pandas as pd

def count_data(train, feature_name):
    feature = train[feature_name].value_counts()
    # feature_name = feature_name[0].upper() + feature_name[1:]
    # type_name = feature_name + ' type'
    feature_df = pd.DataFrame(data={feature_name + ' type': feature.index, 'counts': feature.values})
    print(f'{feature_name} has {len(feature_df)} types, and the counts are:')
    print(f'{feature_df}\n')
    return feature_df

def save_name(tp, loc):
    lang_name = train['name'][loc[0]].unique()
    print(f'{tp} has {len(lang_name)} unique page names')
    zh_df = pd.DataFrame(data={'name': lang_name, 'type': tp})
    zh_df.to_csv(f'./name/{tp[:2]}_name.csv', index=False)

def compute_views(data_df, feature_name):
    # langs = project_df['project type'].to_list()
    types = data_df[feature_name + ' type'].to_list()
    views, percentages = [], []
    for tp in types:
        loc = np.where(train[feature_name] == tp)

        # if tp[3:] == 'wikipedia.org':
        #     save_name(tp, loc)

        # Find the views of given language web pages
        view = np.sum(train['views'][loc[0]])
        views.append(view)

        # Find the percentage of pages that have valid view data
        percentage = np.sum(train['valid_date'][loc[0]] == 550) / len(loc[0])
        percentages.append(round(100 * percentage, 2))


    data_df['total views'] = views
    data_df['average views'] = np.round(views / data_df['counts'], 2)
    data_df['% of valid views'] = percentages
    print(f'{data_df}\n')

    data_df.to_csv(f'./page/{feature_name}.csv', index=False)
    return data_df


if __name__ == '__main__':
    path = os.getcwd()

    # Read page from train_1
    train = pd.read_csv(f'{path}/page/train_1_dates.csv')
    print(f'{train}\n')


    names = train['name'].unique()
    print(f'Name has {len(names)} types\n')

    # Example: find all pages with name 'Beyoncé'
    print('Example: Find all pages with name "Beyoncé"')
    loc = np.where(train['name'] == 'Beyoncé')
    print(f"{train['Page'][loc[0]]}\n")


    print('Stage 1')
    # Count each type of project, access, agent
    access_df = count_data(train, 'access')
    agent_df = count_data(train, 'agent')
    project_df = count_data(train, 'project')


    print('Stage 2')
    access_df = compute_views(access_df, 'access')
    agent_df = compute_views(agent_df, 'agent')
    project_df = compute_views(project_df, 'project')