from matplotlib import pyplot as plt
import numpy as np

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

# Calculate the bin centers
def calc_bin_centers(bins):
    bin_centers = []
    i = 0
    while i < len(bins)-1:
        center = (bins[i+1]+bins[i])/2
        bin_centers.append(center)
        i += 1
    return bin_centers 

# Calculate likelihoods
def calc_likelihoods(hist, centers, testdata):
    i = 0
    likelihoods = []
    while i < len(testdata):
        point = testdata[i]
        # Find closest bin center
        bin_idx = np.argmin(np.abs(np.array(centers)-point))
        # Calculate the likelihood:
        bin_count = hist[bin_idx]
        total_count = sum(hist)
        likelihood = bin_count/total_count
        likelihoods.append(likelihood)
        i += 1
    return likelihoods

# Calculates classification accuracy knowing the true and calculated labels
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

def prob_weigt_height(point, X_train, keyword):
    if keyword == "heigth":
        hist, bins = np.histogram(X_train[:,0], bins=10, range=(80, 220))
        centers = calc_bin_centers(bins)
        bin_idx = np.argmin(np.abs(np.array(centers)-point))
        bin_count = hist[bin_idx]
        total_count = sum(hist)
        return bin_count/total_count
    elif keyword == "weigth":
        hist, bins = np.histogram(X_train[:,1], bins=10, range=(30, 180))
        centers = calc_bin_centers(bins)
        bin_idx = np.argmin(np.abs(np.array(centers)-point))
        bin_count = hist[bin_idx]
        total_count = sum(hist)
        return bin_count/total_count
    elif keyword == "heigth_and_weigth":
        hist, bins = np.histogram(X_train[:,0], bins=10, range=(80, 220))
        centers = calc_bin_centers(bins)
        bin_idx = np.argmin(np.abs(np.array(centers)-point[0]))
        bin_count = hist[bin_idx]
        total_count = sum(hist)
        prob_h = bin_count/total_count
        hist, bins = np.histogram(X_train[:,1], bins=10, range=(30, 180))
        centers = calc_bin_centers(bins)
        bin_idx = np.argmin(np.abs(np.array(centers)-point[1]))
        bin_count = hist[bin_idx]
        total_count = sum(hist)
        prob_w = bin_count/total_count
        return prob_h*prob_w
    else:
        return 0

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

    # Class priors according to training data
    prior_m, prior_f = calc_priors(y_train)
    print("The prior probabilities for male and female are:", prior_m, "and", prior_f)

    # Solve likelihoods
    # Histogram data for female heights
    hist, bins = np.histogram(female_tr[:,0], bins=10, range=(80, 220))
    bin_centers = calc_bin_centers(bins)
    height_female = calc_likelihoods(hist, bin_centers, X_test[:,0]) 
    
    # Histogram data for female weights
    hist, bins = np.histogram(female_tr[:,1], bins=10, range=(30, 180))
    bin_centers = calc_bin_centers(bins)
    weight_female = calc_likelihoods(hist, bin_centers, X_test[:,1]) 

    # Histogram data for male heights
    hist, bins = np.histogram(male_tr[:,0], bins=10, range=(80, 220))
    bin_centers = calc_bin_centers(bins)
    height_male = calc_likelihoods(hist, bin_centers, X_test[:,0]) 

    # Histogram data for male weights
    hist, bins = np.histogram(male_tr[:,1], bins=10, range=(30, 180))
    bin_centers = calc_bin_centers(bins)
    weight_male = calc_likelihoods(hist, bin_centers, X_test[:,1])

    # Classify by height
    labels = []
    i = 0
    while i < len(X_test):
        # Calculate p(A|B) = p(A)*p(B|A)/p(B)
        # p(heigth) can be calculated from the histogram of the whole training data,
        # but it's not necessarily needed since it doesn't effect the maximum likelihood decision
        p_heigth = prob_weigt_height(X_test[i,0], X_train, "heigth")
        male = prior_m * height_male[i] / p_heigth
        female = prior_f * height_female[i] / p_heigth
        if male > female:
            labels.append(0)
        else:
            labels.append(1)
        i += 1
    accuracy = calc_accuracy(labels, y_test)   
    print("Classification accuracy based on heigth is:", accuracy)

    # Classify by weight
    labels = []
    i = 0
    while i < len(X_test):
        p_weigth = prob_weigt_height(X_test[i,1], X_train, "weigth")
        male = prior_m * weight_male[i] / p_weigth
        female = prior_f * weight_female[i] / p_weigth
        if male > female:
            labels.append(0)
        else:
            labels.append(1)
        i += 1
    accuracy = calc_accuracy(labels, y_test)   
    print("Classification accuracy based on weigth is:", accuracy)

    # Classify by height and weight
    labels = []
    i = 0
    while i < len(X_test):
        prob_hw = prob_weigt_height(X_test[i,:], X_train, "heigth_and_weigth")
        male = prior_m * height_male[i]*weight_male[i] / prob_hw
        female = prior_f * height_female[i]*weight_female[i] / prob_hw
        if male > female:
            labels.append(0)
        else:
            labels.append(1)
        i += 1
    accuracy = calc_accuracy(labels, y_test)   
    print("Classification accuracy based on heigth and weight is:", accuracy)

    # Classification accuracy with height and weigth is the highest, which seems probable
    
main()

