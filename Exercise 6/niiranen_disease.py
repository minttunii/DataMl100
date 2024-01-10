import numpy as np
import sklearn.metrics
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor

def main():
    Xtrain = np.loadtxt("Exercise 6/disease_X_train.txt")
    ytrain = np.loadtxt("Exercise 6/disease_y_train.txt")
    Xtest = np.loadtxt("Exercise 6/disease_X_test.txt")
    ytest= np.loadtxt("Exercise 6/disease_y_test.txt")

    # a) Baseline
    # Calculate training data mean value (disease severity)
    ypred = [np.mean(ytrain)] * len(ytest)
    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    print("Baseline MSE error:", mse)

    # b) Linear model
    lin_model = LinearRegression()
    # Fit model
    lin_model.fit(Xtrain, ytrain)

    # Predict labels for test data
    ypred = lin_model.predict(Xtest)
    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    print("Linear model MSE error:", mse)

    # c) Decision tree reggressor
    dec_tree = DecisionTreeRegressor(criterion="squared_error")
    # Fit training data to model
    dec_tree.fit(Xtrain, ytrain)
    # export_graphviz(dec_tree, out_file ='tree.dot')
    ypred = dec_tree.predict(Xtest)
    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    print("Decision tree reggressor MSE error:", mse)

    # d) Random forest reggressor
    rand_forest = RandomForestRegressor(criterion="squared_error")
    # Fit model
    rand_forest.fit(Xtrain, ytrain)
    ypred = rand_forest.predict(Xtest)
    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    print("Random forest reggressor MSE error:", mse)

    # Linear model and random forest reggressor seem to have the smallest MSE

main()
