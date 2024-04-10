from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np 
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import myLogger

def readData():
    logObj  = myLogger.giveMeLoggingObject()
    iris = datasets.load_iris()
    print(type(iris.data), type(iris.target))

    # We want to check what kind of data is being loaded in to make sure it is
    # not erronous data that is the result of a poisoning attack.
    logObj.info('Data types read by file:\n\t %s\n', type(iris.data))
    
    X = iris.data
    Y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    print(df.head())
    
    # Since poisoning attacks focus on injecting erronous data into the dataset,
    # I am going to log the data that is being read by the ML code.
    logObj.info('Dataset file to be read:\n %s\n', df.head())

    return df 

def makePrediction():
    logObj  = myLogger.giveMeLoggingObject()
    
    iris = datasets.load_iris()

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(iris['data'], iris['target'])
    X = [
        [5.9, 1.0, 5.1, 1.8],
        [3.4, 2.0, 1.1, 4.8],
    ]
    prediction = knn.predict(X)
    print(prediction)

    # Since model tricking occurs when models are attacked and cause them to make
    # erronous results/affect prediction performace, I am going to log the
    # predictions that are made by the ML code.
    logObj.info('Prediction made by algorithm: %s\n', prediction)

def doRegression():
    diabetes = datasets.load_diabetes()
    
    logObj  = myLogger.giveMeLoggingObject()
    
    # We want to check what kind of data is being loaded in to make sure it is
    # not erronous data that is the result of a poisoning attack.
    logObj.info("Dataset loaded into Regression Function: \n\t %s\n", diabetes)

    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # Since model tricking occurs when models are attacked and cause them to make
    # erronous results/affect prediction performace, I am going to log the
    # predictions that are made by the ML code.
    logObj.info('Prediction made by algorithm: %s\n', diabetes_y_pred)


def doDeepLearning():
    logObj  = myLogger.giveMeLoggingObject()
    
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()


    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5


    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    num_filters = 8
    filter_size = 3
    pool_size = 2

    model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # Train the model.
    model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels)),
    )

    # Since poisoning attacks focus on injecting erronous data into the dataset,
    # I am going to log the data that is being read by the ML code.
    logObj.info('Validation data for deep learning:\n\t %s\n', validation_data)

    model.save_weights('cnn.h5')

    predictions = model.predict(test_images[:5])

    print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

    # Since model tricking occurs when models are attacked and cause them to make
    # erronous results/affect prediction performace, I am going to log the
    # predictions that are made by the ML code.
    logObj.info('Prediction made by deep learning model: \t%s\n', predictions)

    print(test_labels[:5]) # [7, 2, 1, 0, 4]


if __name__=='__main__': 
    data_frame = readData()
    makePrediction() 
    doRegression() 
    doDeepLearning() 
