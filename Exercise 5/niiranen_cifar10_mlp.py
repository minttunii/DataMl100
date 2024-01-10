import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def load_data_and_labels(file):
    datadict = unpickle(file)
    Xtr = datadict["data"]
    Ytr = datadict["labels"]
    # RGB pixel values are between [0, 255] so we can use this to normalize data
    Xtr = Xtr/255
    return Xtr, Ytr

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

def main():
    # Load all training sets 1-5
    Xtr1, Ytr1 = load_data_and_labels("Exercise 5/cifar-10-python/data_batch_1")
    Xtr2, Ytr2 = load_data_and_labels("Exercise 5/cifar-10-python/data_batch_2")
    Xtr3, Ytr3 = load_data_and_labels("Exercise 5/cifar-10-python/data_batch_3")
    Xtr4, Ytr4 = load_data_and_labels("Exercise 5/cifar-10-python/data_batch_4")
    Xtr5, Ytr5 = load_data_and_labels("Exercise 5/cifar-10-python/data_batch_5")

    # Combine training datasets to one training set
    Xtr = np.vstack((Xtr1, Xtr2, Xtr3, Xtr4, Xtr5))
    Ytr = np.hstack((Ytr1, Ytr2, Ytr3, Ytr4, Ytr5))
    Ytr = keras.utils.to_categorical(Ytr, num_classes=10)

    # Label names for the 10 classes
    labeldict = unpickle("Exercise 5/cifar-10-python/batches.meta")
    label_names = labeldict["label_names"]

    # Load test data
    Xtest, Ytest1 = load_data_and_labels("Exercise 5/cifar-10-python/test_batch")
    Ytest = keras.utils.to_categorical(Ytest1, num_classes=10)

    model = Sequential()
    n_epochs = 40
    # First layer of 20 neurons
    model.add(Dense(40, input_dim=3072, activation='sigmoid'))
    # Dropout layer
    model.add(keras.layers.Dropout(0.2))
    # Last layer of 10 neurons
    model.add(Dense(10, input_dim=5, activation='sigmoid'))
    model.summary()

    # Lets try to train the model
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.5) # Gradient decsent optimizer
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model_history = model.fit(Xtr, Ytr, batch_size=100, epochs=n_epochs, verbose=1)

    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(model_history.history['loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training'], loc='upper right')

    # Plot training accuracy
    plt.subplot(2, 1, 2)
    plt.plot(model_history.history['accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training'], loc='upper left')
    plt.show()

    # Withfirst two layers (5 input neurons), 10 epochs and learning rate of 0.01, we get around 33% accuracy for training
    # The loss function categorical crossentropy and activation function sigmoid is used 

    # Lets increase the number of neurons in the input layer to 10 and epochs to 15
    # This changed toheresults to loss: 1.7330 - accuracy: 0.3867 after the last epoch, so the accuracy increased

    # Lets add dropout layer to the model with rate 0.2
    # This changed results to loss: 1.8815 - accuracy: 0.3142 after the last epoch, so the accuracy decreased
    # Lets adjust the other parameters to get more accuracy

    # Lets increase the number of neurons in the input layer and epochs to 20
    # This changed the results to loss: 1.7745 - accuracy: 0.3667 after the last epoch, so the accuracy increased

    # Lets try different learning rate, for example 0.1 to the gradient decsent optimizer
    # This changed the results to loss: 1.6782 - accuracy: 0.3935 after the last epoch, so the accuracy increased little
    # Bigger learning rate like 0.5 decreased the results to loss: 1.7521 - accuracy: 0.3617, so 0.1 seems more suitable

    # Lets increase the number of neurons in the input layer and epochs to 30
    # This changed the results to loss: 1.6615 - accuracy: 0.4039 after the last epoch, so the accuracy increased

    # Batch size for model.fit() can also be changed, the default value is 32
    # Batch size 64 gives loss: 1.6190 - accuracy: 0.4179
    # Batch size 100 gives loss: 1.6100 - accuracy: 0.4231

    # Lets increase the number of neurons in the input layer and epochs to 30
    # This changed the results to loss: 1.5071 - accuracy: 0.4573, so the accuracy increased

    # Lets apply our model to the test data and print accuracy and loss via model.evaluation()
    test_results = model.evaluate(Xtest, Ytest, verbose=1)
    print()
    print('Test loss:', test_results[0])
    print('Test accuracy:', test_results[1])
    # Lets also calculate the classification accuracy via probabilities from model.prediction()
    y_prob = model.predict(Xtest)
    y_classes = y_prob.argmax(axis=-1)
    accuracy = calc_accuracy(y_classes, Ytest1)
    print("Test accuracy:", accuracy)

    # Print classification accuracy and loss also for training data
    print()
    print("Training loss:", model_history.history['loss'][n_epochs-1])
    print("Training accuracy:", model_history.history['accuracy'][n_epochs-1])


main()

