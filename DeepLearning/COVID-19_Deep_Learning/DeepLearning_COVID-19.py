# ..:Zorica~Koceva:.. #
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, Conv3D, LSTM, GRU, MaxPooling2D, Flatten
from keras.preprocessing import image
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
    _______________________________________________________________________________________________

    Defining paths for folders where I'll put digital images od CT scans for Training and Test sets
    -------
    source: X rays normal healthy lungs && x rays COVID-19 lungs
    -------
    https://www.google.com/search?q=x+rays+normal+healthy+lungs&sxsrf=ALeKk01cBs8SIp5KtflYYnstTr91YjMFUg:1596529611608&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj_6v2ikIHrAhUjyIUKHfoWB_QQ_AUoAXoECAwQAw&biw=1536&bih=674
    ________________________________________________________________________________________________
"""
TRAINING_FOLDER_PATH = "COVID_Dataset/Training"
TEST_FOLDER_PATH = "COVID_Dataset/Test"

if __name__ == '__main__':

    """
        Models:
        For one input it's better to use Sequential Keras model
        For 2++ (two or more) inputs, we can use Model Keras model
    """

    """
        Layers:
        *Dense - regular deeply connected neural network layer
        *Dropout - regularization.. Easily implemented by randomly selecting nodes to be dropped-out with a given probability 
        *Conv1D, Conv2D, Conv3D - Convolution Layers 
        *LSTM - Recurrent Layer
        *GRU - Recurrent Integration version Layer
        *MaxPooling2D - Max pooling to a convolutional neural network in code
        *Flatten - used to reshape the tensor to such a shape which is equal to the number of elements present in the tensor

    """

    # Training model from one input (CT scan picture)
    model_CT_lungs = Sequential()
    model_CT_lungs.add(Conv2D(32, kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=(224, 224, 3)))
    model_CT_lungs.add(Conv2D(128, (3, 3), activation='relu'))
    model_CT_lungs.add(MaxPooling2D(pool_size=(2, 2)))
    model_CT_lungs.add(Dropout(0.25))

    model_CT_lungs.add(Conv2D(64, (3, 3), activation='relu'))
    model_CT_lungs.add(MaxPooling2D(pool_size=(2, 2)))
    model_CT_lungs.add(Dropout(0.25))

    model_CT_lungs.add(Conv2D(128, (3, 3), activation='relu'))
    model_CT_lungs.add(MaxPooling2D(pool_size=(2, 2)))
    model_CT_lungs.add(Dropout(0.25))

    model_CT_lungs.add(Flatten())
    model_CT_lungs.add(Dense(64, activation='relu'))
    model_CT_lungs.add(Dropout(0.5))
    model_CT_lungs.add(Dense(1, activation='sigmoid'))

    """
        The last layer is Output Layer, should have same number of neurons like number of classes (2:Healthy and COVID-19 diseased lungs)
        Softmax function is used for classification, because to the layer gives sum of neurons to be equal to 1 (or mathematical said
        give us probabilities for each of the classes )

    """

    """
        Using Adam optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
        to update network weights iterative based in training data.
        Means that a single learning rate for all weight updates and the learning rate does not change during training.
    """

    optimizer_lungs_COVID = Adam()
    model_CT_lungs.compile(loss=binary_crossentropy,
                           optimizer=optimizer_lungs_COVID,
                           metrics=['accuracy'])

    # Getting all the parameters
    model_CT_lungs.summary()

    # Moulding train images
    training_data_generator = image.ImageDataGenerator(rescale=1. / 255,
                                                       shear_range=0.2,
                                                       zoom_range=0.2,
                                                       horizontal_flip=True)  # lungs shape is vertical, I don't need horizontal images

    test_dataset = image.ImageDataGenerator(rescale=1. / 255)

    # Reshaping both test and validation images from data sets
    training_set_generator = training_data_generator.flow_from_directory('COVID_Dataset/Training',
                                                                         target_size=(224, 224),
                                                                         batch_size=32,
                                                                         class_mode='binary')

    test_set_generator = test_dataset.flow_from_directory('COVID_Dataset/Test',
                                                           target_size=(224, 224),
                                                           batch_size=32,
                                                           class_mode='binary')


    # Training the model
    covid_model_lungs_training = model_CT_lungs.fit_generator(training_set_generator,
                                                              steps_per_epoch=10,
                                                              epochs=15,
                                                              validation_data=test_set_generator,
                                                              validation_steps=2)

    # Getting summary results from CT lungs model from training data set
    scanCT_results = covid_model_lungs_training.history
    print(scanCT_results)

    model_CT_lungs.save("COVID-19_CTlungs_model.h5")

    model_CT_lungs.evaluate_generator(training_set_generator)

    evaluating_test_generator = model_CT_lungs.evaluate_generator(test_set_generator)
    print(evaluating_test_generator)

    # Confusion Matrix for data from CT lungs scan

    training_set_generator.class_indices
    y_actual, y_test = [], []

    # First, I will work with Test Set folder with CT scans of healthy lungs
    for i in os.listdir("./COVID_Dataset/Test/Healthy/"):
        scanCT = image.load_img("./COVID_Dataset/Test/Healthy/" + i, target_size=(224, 224))
        scanCT = image.img_to_array(scanCT)
        scanCT = np.expand_dims(scanCT, axis=0)  # numpy transformed

        prediction_lungs_healthy = model_CT_lungs.predict_classes(scanCT)

        y_test.append(prediction_lungs_healthy[0, 0])
        y_actual.append(1)

    # And then will work with Test Set folder with CT scans of COVID-19_diseased lungs
    for i in os.listdir("./COVID_Dataset/Test/COVID-19_diseased/"):
        scanCT = image.load_img("./COVID_Dataset/Test/COVID-19_diseased/" + i, target_size=(224, 224))
        scanCT = image.img_to_array(scanCT)
        scanCT = np.expand_dims(scanCT, axis=0)

        prediction_lungs_diseased = model_CT_lungs.predict_classes(scanCT)
        y_test.append(prediction_lungs_diseased[0, 0])
        y_actual.append(0)

    y_actual = np.array(y_actual)
    y_test = np.array(y_test)


    cn = confusion_matrix(y_actual, y_test)

    sns.heatmap(cn, cmap="plasma", annot=True)
    """
    Explanation:
        if False, True
        0: Covid ; 1: Normal
    """
