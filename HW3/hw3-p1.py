import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# import ipdb
# ipdb.set_trace()

def fisher_score(x, y):
    mean = np.mean(x, axis=0)

    # Sector
    c0 = y == 0
    c1 = y == 1

    # Compute fisher score
    num = len(x[c0]) * (np.mean(x[c0], axis=0) - mean) ** 2 + len(x[c1]) * (np.mean(x[c1], axis=0) - mean) ** 2
    den = len(x[c0]) * np.var(x[c0], axis=0) + len(x[c1]) * np.var(x[c1], axis=0)
    fisher = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
    
    return fisher


if __name__ == '__main__':
    ### Load data ###
    indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)
    x = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T
    y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
    y = (y>0).astype(int).reshape(y.shape[0])


    ### Feature ranking ###
    # 1. Random
    # ranking_idx = np.arange(x.shape[1])
    # random.shuffle(ranking_idx)

    # 2. sklearn Fisher score
    # from sklearn.feature_selection import f_classif
    # ranking_idx = f_classif(x, y)[0]
    # ranking_idx = np.argsort(ranking_idx)[::-1]

    # 3. Fisher score without sklearn
    ranking_idx = fisher_score(x, y)
    ranking_idx = np.argsort(ranking_idx)[::-1]


    ### Feature evaluation ###
    # Use a simple dicision tree with 5-fold validation to evaluate the feature selection result.
    # You can try other classifier and hyperparameter.
    classif = ['Decision Tree', 'SVM']
    for i in classif:
        print(i)
        score_history = []
        for m in range(5, 2001, 5):
            # Select Top m feature
            x_subset = x[:, ranking_idx[:m]]

            # Build random forest
            if i == 'Decision Tree':
                clf = DecisionTreeClassifier(random_state=0)
            else:
                clf = SVC(kernel='rbf', random_state=0) #build SVM

            # Calculate validation score
            scores = cross_val_score(clf, x_subset, y, cv=5)

            # Save the score calculated with m feature
            score_history.append(scores.mean())

        # Report best accuracy.
        print(f"Max accuracy: {max(score_history)}")
        print(f"Number of features: {np.argmax(score_history)*5+5}")
        
        # Show the selected features
        feature_idx = ranking_idx[:np.argmax(score_history)*5+5]
        print(f"Selected features:\n{np.array(indexes)[feature_idx, 0]}\n")

        ### Visualization ###
        plt.plot(range(5, 2001, 5), score_history)
    
    plt.title('Original')
    plt.xlabel('Number of features')
    plt.ylabel('Cross-validation score')
    plt.legend(classif)
    # plt.savefig('./images/hw3-1_result.png')
    plt.show()