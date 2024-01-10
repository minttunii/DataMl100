import numpy as np
import random

# Function assigns a random class from given options
def random_class():
    classes = [0.0, 1.0]
    return random.choice(classes)

def calc_accuracy(labels, true_labels):
    N_true = 0
    N_false = 0
    i = 0
    while i < len(labels):
        if labels[i] == true_labels[i]:
            N_true += 1
        else:
            N_false += 1
        i += 1
    accuracy = N_true / (N_true + N_false)
    return accuracy

# Calculates the prior probabilities for classes male and female from training data
def calc_priors(true_labels):
    n = len(true_labels)
    n_female = 0
    n_male = 0
    for x in true_labels:
        if x == 0:
            n_male += 1
        else:
            n_female += 1
    prior_f = n_female/n
    prior_m = n_male/n
    return prior_m, prior_f

def main():
    # Read data files
    y_train = np.loadtxt("Exercise 4/male_female_y_train.txt") # Label 0 male, 1 female
    X_train = np.loadtxt("Exercise 4/male_female_X_train.txt") # Hight, weight colums
    y_test = np.loadtxt("Exercise 4/male_female_y_test.txt") # Label 0 male, 1 female
    X_test = np.loadtxt("Exercise 4/male_female_X_test.txt") # Hight, weight colums

    # Classify test cases randomly
    rand_labels = []
    i = 0
    while i < len(X_test):
        rand_labels.append(random_class())
        i += 1
    rand_labels = np.array(rand_labels)
    accuracy = calc_accuracy(rand_labels, y_test)
    print("Accuracy of the random classifier is:", accuracy)

    # Classify all test cases to the most likely class
    prior_m, prior_f = calc_priors(y_train)
    ml_class = ""
    
    if(prior_m > prior_f):
        # All test cases are assigned to class male
        labels = [0] * len(y_test)
        ml_class = "male"
    else:
        # All test cases are assigned to female
        labels = [1] * len(y_test)
        ml_class = "female"

    # Print accuracy
    labels = np.array(labels)
    accuracy = calc_accuracy(labels, y_test)
    print("Accuracy when all test cases are assigned to class", ml_class, "is:", accuracy)

main()
