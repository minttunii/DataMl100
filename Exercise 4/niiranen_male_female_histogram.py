import matplotlib.pyplot as plt
import numpy as np


def main():
    # Read data files
    y_train = np.loadtxt("Exercise 4/male_female_y_train.txt") # Label 0 male, 1 female
    X_train = np.loadtxt("Exercise 4/male_female_X_train.txt") # Hight, weight colums
    y_test = np.loadtxt("Exercise 4/male_female_y_test.txt") # Label 0 male, 1 female
    X_test = np.loadtxt("Exercise 4/male_female_X_test.txt") # Hight, weight colums
    female_tr = []
    male_tr = []
    i = 0
    for label in y_train:
        if label == 0:
            # Male
            male_tr.append(X_train[i,:])
        else:
            # Female
            female_tr.append(X_train[i,:])
        i = i + 1
    female_tr = np.array(female_tr, dtype=float)
    male_tr = np.array(male_tr, dtype=float)

    # Histograms for height
    hist, bins = np.histogram(female_tr[:,0], bins=10, range=(80, 220))
    plt.hist(bins[:-1], bins, weights=hist, edgecolor="red", label="Female")
    hist, bins = np.histogram(male_tr[:,0], bins = 10, range=(80, 220))
    plt.hist(bins[:-1], bins, weights=hist, edgecolor="blue", label="Male")
    plt.legend()
    plt.xlabel("Heights/cm")
    plt.show()

    # Histograms for weigh
    hist, bins = np.histogram(female_tr[:,1], bins=10, range=(30, 180))
    plt.hist(bins[:-1], bins, weights=hist, edgecolor="red", label="Female")
    hist, bins = np.histogram(male_tr[:,1], bins=10, range=(30, 180))
    plt.hist(bins[:-1], bins, weights=hist, edgecolor="blue", label="Male")
    plt.legend()
    plt.xlabel("Weights/kg")
    plt.show()

    # Visually, height seems little better for classification, the weight histograms are overlapping more

main()
