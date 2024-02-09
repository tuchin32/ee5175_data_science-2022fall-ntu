import numpy as np
import pandas as pd
import random
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

### Feature evaluation ###
# Use a simple dicision tree with 5-fold validation to evaluate the feature selection result.
# You can try other classifier and hyperparameter.
def fitness_function(data, individual):
    # Select Top m feature
    feature_idx = np.where(np.array(individual) == 1)[0]
    data_subset = data[:, feature_idx]

    # Build random forest
    clf = DecisionTreeClassifier(random_state=0)
    # clf = SVC(kernel='rbf', random_state=0) #build SVM

    # Calculate validation score
    scores = cross_val_score(clf, data_subset, y, cv=5)

    # Save the score calculated with m feature
    return scores.mean()

def selection(population, fitness, population_size, type='roulette'):
    if type == 'roulette':
        # Select the parents based on roulette wheel selection
        fitness = np.array(fitness)
        fitness = fitness / fitness.sum()
        parent_idx = np.random.choice(population_size, size=population_size, p=fitness)
        parents = [population[i] for i in parent_idx.tolist()]
    elif type == 'tournament':
        # Select the parents based on tournament selection
        parents = []
        for i in range(population_size):
            idx = np.random.choice(population_size, size=2, replace=False)
            if fitness[idx[0]] > fitness[idx[1]]:
                parents.append(population[idx[0]])
            else:
                parents.append(population[idx[1]])
    else:
        fitness_idx = np.argsort(fitness)
        parents = [population[idx] for idx in fitness_idx[:population_size // 2]]
        parents += [population[idx] for idx in fitness_idx[:population_size // 2]]

    return parents

def genetic_algorithm(data, bit_length, generation, population_size, p_cross=0.8, p_mut=0.1):
    '''
    population: (population_size, bit_length)
    fitness: (population_size, )
    parents: (population_size, bit_length)
    children: (population_size, bit_length)

    best individual: (generation, bit_length)
    best fitness: (generation, )
    '''

    # Initialize population
    population = np.random.randint(2, size=(population_size, bit_length))
    population = population.tolist()

    # Track the best
    best_individual = []
    best_fitness = []

    # Start evolution
    for gen in range(generation + 1):
        if gen % 10 == 0:
            print(f'Generation: {gen}')
        # Evaluate the fitness of each individual
        fitness = []
        for individual in population:
            fitness.append(fitness_function(data, individual))

        # Save the best individual
        best_individual.append(population[np.argmax(fitness)])
        best_fitness.append(np.max(fitness))
        
        # Select the parents based on roulette wheel selection
        parents = selection(population, fitness, population_size, type='roulette')

        # Crossover
        children = parents 
        for i in range(0, population_size, 2):
            if random.random() < p_cross:
                crossover_point = random.randint(1, bit_length - 1)
                children[i][crossover_point:], children[i + 1][crossover_point:] = parents[i + 1][crossover_point:], parents[i][crossover_point:]

        # Mutation
        for i in range(population_size):
            if random.random() < p_mut:
                mutation_point = random.randint(0, bit_length - 1)
                children[i][mutation_point] = 1 - children[i][mutation_point]

        # Replace the old population with the new one
        population = children
        population[0] = best_individual[-1]

    return best_individual, best_fitness


if __name__ == '__main__':
    ### Load data ###
    indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)
    x = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T
    y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
    y = (y>0).astype(int).reshape(y.shape[0])


    ### Feature ranking ###
    # Fisher score without sklearn
    ranking_idx = fisher_score(x, y)
    ranking_idx = np.argsort(ranking_idx)[::-1]


    ### Subset-Based Feature Selection ###
    # Use genetic algorithm to select the best subset of features
    x_truncated = x[:, ranking_idx[:100]]
    bit_length = x_truncated.shape[1]
    generation = 100
    population_size = 100
    feature_history, score_history = genetic_algorithm(x_truncated, bit_length, generation, population_size, 0.8, 0.03)
    num_features = np.sum(feature_history, axis=1)

    # Report best accuracy and the selected features.
    max_idx = np.argmax(score_history)
    print(f'Best result occurs in {max_idx}-th generation.')
    print(f"Max of Decision Tree: {score_history[max_idx]}")
    print(f"Number of features: {num_features[max_idx]}")
    

    ### Visualization ###
    plt.plot(np.arange(len(score_history)), score_history, c='blue')
    plt.title('Subset-Based Feature Selection')
    plt.xlabel('Generation')
    plt.ylabel('Cross-validation score')
    plt.legend(['Decision Tree'])
    # plt.savefig('./images/hw3-2_result.png')
    plt.show()