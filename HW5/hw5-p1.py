import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

# Use two-way split gini index to build a decision tree
class DecisionTree:
    def __init__(self, data, columns = None, depth = 0, max_depth = 10):
        self.data = data
        self.columns = columns
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.gini_candidates = {'Feature': [], 'Value': [], 'Gini Index': []}
        self.best_gini = None
        self.split_feature = None
        self.split_value = None
        self.target = None
        self.build_tree()
    
    def build_tree(self):
        # If the data is empty, return
        if self.data.empty:
            return
        
        # If the data is pure, return
        if len(self.data[self.columns[-1]].unique()) == 1:
            self.target = self.data[self.columns[-1]].unique()[0]
            return
        
        # If the depth is greater than the max depth, return
        if self.depth >= self.max_depth:
            self.target = self.data[self.columns[-1]].value_counts().idxmax()
            return
        
        # Get the best split feature and value
        self.best_gini, self.split_feature, self.split_value = self.get_best_split()
        
        # Split the data
        left_data = self.data[self.data[self.split_feature].isin(self.split_value)]
        right_data = self.data[~self.data[self.split_feature].isin(self.split_value)]
        
        # Display the decision tree
        self.display_tree(left_data, right_data)

        # Save the children nodes to csv file
        left_data.to_csv(f'left_node_{self.depth}.csv', index = False)
        right_data.to_csv(f'right_node_{self.depth}.csv', index = False)

        # Build the left and right subtrees
        self.left = DecisionTree(left_data, self.columns, self.depth + 1, self.max_depth)
        self.right = DecisionTree(right_data, self.columns, self.depth + 1, self.max_depth)

    
    def display_tree(self, left_data, right_data):
        print('Depth:', self.depth)

        candidates = pd.DataFrame(self.gini_candidates)
        candidates = candidates.sort_values(by = 'Gini Index', ascending = True)
        print(f'Gini candidates:\n{candidates}')

        split = pd.DataFrame()
        split['Split Feature'] = [self.split_feature]
        split['Split Value'] = [self.split_value]
        split['Gini Index'] = [self.best_gini]
        
        print(f'Split parameter:\n{split}')
        print(f'Left node:\n{left_data}')
        print(f'Right node:\n{right_data}\n')

    def get_best_split(self):
        # Get the best split feature and value
        best_split_feature = None
        best_split_value = None
        best_gini = 1.0
        
        # Loop through all the features
        for feature in self.columns[1:-1]:
            # Get the unique values of the feature
            values = self.data[feature].unique()
            # print('ori', values)

            # Build up the combination of the values
            new_values = []
            for i in range(len(values) // 2):
                new_values.extend(list(combinations(values, i + 1)))
            
            # Loop through all the values
            for value in new_values:
                # Split the data
                left_data = self.data[self.data[feature].isin(value)]
                right_data = self.data[~(self.data[feature].isin(value))]
                
                # Calculate the gini index
                gini = self.get_gini(left_data, right_data)
                self.gini_candidates['Feature'].append(feature)
                self.gini_candidates['Value'].append(value)
                self.gini_candidates['Gini Index'].append(round(gini, 4))
                
                # If the gini index is less than the best gini index, update the best gini index
                if gini < best_gini:
                    best_gini = round(gini, 4)
                    best_split_feature = feature
                    best_split_value = value
        
        # Return the best split feature and value
        return best_gini, best_split_feature, best_split_value

    def get_gini(self, left_data, right_data):
        # Calculate the gini index
        gini_left = 0.0
        gini_right = 0.0
        
        # Get the target categories
        target = self.data[self.columns[-1]].unique()
        
        # Calculate the gini index
        if len(left_data) > 0:
            for t in target:
                gini_left += (len(left_data[left_data[self.columns[-1]] == t]) / len(left_data)) ** 2
        gini_left = 1 - gini_left

        if len(right_data) > 0:
            for t in target:
                gini_right += (len(right_data[right_data[self.columns[-1]] == t]) / len(right_data)) ** 2
        gini_right = 1 - gini_right

        gini = (len(left_data) / len(self.data)) * gini_left + (len(right_data) / len(self.data)) * gini_right

        return gini


if __name__ == '__main__':
    # Read in the data
    df = pd.read_csv('data.csv')
    print(df)

    # Get the column names
    columns = df.columns
    print(columns)

    # Get the unique feature categories and the target categories
    gender = df[columns[1]].unique()
    car = df[columns[2]].unique()
    shirt = df[columns[3]].unique()
    target = df[columns[4]].unique()

    # Build the decision tree
    tree = DecisionTree(df, columns)


    


    