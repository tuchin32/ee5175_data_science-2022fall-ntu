import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from wordcloud import WordCloud

def plot_pie_chart(data, data_type, colors):
    types = data[f'{data_type} type'].tolist()
    sizes = data['counts'].tolist()
    labels = [f'{label}\n({size})' for label, size in zip(types, sizes)]
    explode = [0.01] * len(labels)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, counterclock=False)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(f'{data_type[0].upper() + data_type[1:]}: counts and ratio of each type', pad=20)

    #draw circle
    centre_circle = plt.Circle((0, 0), 0.75, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.savefig(f'./page/page_image/{data_type}_pie_chart.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def plot_bar_chart(data, data_type, colors):
    labels = data[f'{data_type} type'].tolist()
    if data_type == 'project':
        labels = [label.split('.')[0] for label in labels]
    sizes = data['average views'].tolist()
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, sizes, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    x_label = f'{data_type[0].upper() + data_type[1:]}'
    ax.set_title(f'{x_label}: average views of each type', pad=20)
    ax.set_xlabel(f'{x_label} type')
    ax.set_ylabel('Average views')

    plt.savefig(f'./page/page_image/{data_type}_bar_chart.jpg', dpi=300, bbox_inches='tight')
    plt.show()

# Plot 100% stacked bar chart
def plot_stacked_bar_chart(data, data_type, colors='#9ba88d'):
    labels = data[f'{data_type} type'].tolist()
    if data_type == 'project':
        labels = [label.split('.')[0] for label in labels]
    x_pos = np.arange(len(labels))
    x_label = f'{data_type[0].upper() + data_type[1:]}'
    y_label = 'Percentage'

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(f'{x_label}: percentage of valid views data', pad=20)
    ax.set_xlabel(f'{x_label} type')
    ax.set_ylabel(y_label)

    valid_percent = data['% of valid views'].tolist()
    invalid_percent = list(100 - np.array(valid_percent))

    # Plot the stacked bar chart
    ax.bar(x_pos, valid_percent, color=colors)
    ax.bar(x_pos, invalid_percent, bottom=valid_percent, color='#555647')
    ax.legend(labels=['Valid', 'Invalid'])

    # Plot the percentage on top of each bar
    for i in range(len(labels)):
        ax.text(x_pos[i] - 0.2, valid_percent[i] / 2, f'{valid_percent[i]}%', color='black')
        ax.text(x_pos[i] - 0.2, valid_percent[i] + invalid_percent[i] / 2, f'{round(invalid_percent[i], 2)}%', color='white')
    
    plt.savefig(f'./page/page_image/{data_type}_stacked_bar_chart.jpg', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    path = os.getcwd()

    # Read project, access, agent
    project = pd.read_csv(f'{path}/page/project.csv')
    access = pd.read_csv(f'{path}/page/access.csv')
    agent = pd.read_csv(f'{path}/page/agent.csv')

    # Plot pie chart
    colors = ['#beb9b6', '#d4b192', '#a16d5d', '#caa9a2', '#856b5a', '#af9b8d', '#838a92', '#a5a4aa', '#c2cccd']
    plot_pie_chart(project, 'project', colors[:len(project)])
    plot_pie_chart(access, 'access', colors[:len(access)])
    plot_pie_chart(agent, 'agent', colors[:len(agent)])

    # # Plot bar chart
    # plot_bar_chart(project, 'project', colors[:len(project)])
    # plot_bar_chart(access, 'access', colors[:len(access)])
    # plot_bar_chart(agent, 'agent', colors[:len(agent)])

    # # Plot stacked bar chart
    # plot_stacked_bar_chart(project, 'project')
    # plot_stacked_bar_chart(access, 'access')
    # plot_stacked_bar_chart(agent, 'agent')
