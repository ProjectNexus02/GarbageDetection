from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from sklearn.model_selection import train_test_split
import os
import cv2
import pandas as pd
import numpy as np

from ..dataset.csv_path import CSV_PATH, IMGS_PATH


class Trainer:
    TARGET_HEIGHT = 300
    TARGET_WIDTH = 300

    def __init__(self):
        pass

    def load_garbage_detection_model(self):
        model = load_model("garbage_classification.h5")

    def train_model(self):
        train = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, validation_split=0.1,
                                   rescale=1./255, shear_range=0.1, zoom_range=0.1, width_shift_range=0.1,
                                   height_shift_range=0.1)
        test = ImageDataGenerator(rescale=1./255, validation_split=0.1)
        train_generator = train.flow_from_directory(IMGS_PATH, target_size=(Trainer.TARGET_HEIGHT, Trainer.TARGET_WIDTH),
                                                    batch_size=32, class_mode="categorical", subset="training")
        test_generator = train.flow_from_directory(IMGS_PATH, target_size=(Trainer.TARGET_HEIGHT, Trainer.TARGET_WIDTH),
                                                   batch_size=32, class_mode="categorical", subset="validation")
        labels = {v: k for k, v in train_generator.class_indices.items()}
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=(Trainer.TARGET_HEIGHT, Trainer.TARGET_WIDTH, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(6, activation="softmax"))

        filepath = "garbage_classification.h5"
        checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint1]
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276 // 32, validation_data=test_generator,
                            validation_steps=251 // 32, callbacks=callbacks_list)

    # def train(self):
    #     self.__one_hot_encoding()

    @staticmethod
    def __model(input_shape, output_size):
        model = Sequential()
        model.add(Dense(16, input_shape=input_shape, activation="relu"))
        model.add(Dense(output_size, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    #
    # def __one_hot_encoding(self):
    #     df = pd.read_csv(CSV_PATH)
    #     categories = np.array(list(set(df.garbage_type)))
    #     height = df.height[0]
    #     width = df.width[0]
    #     # num_pixels = height * width
    #     X, y = df.path, df.garbage_type
    #     # X_labels = self.__get_one_hot_encoding(np.array(y), categories)
    #     # X = np.array([cv2.imread(filepath) for filepath in X])
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.2, random_state=42, shuffle=True
    #     )
    #     train_labels = np.array(y_train)
    #     test_labels = np.array(y_test)
    #     X_train_labels = self.__get_one_hot_encoding(train_labels, categories)
    #     X_test_labels = self.__get_one_hot_encoding(test_labels, categories)
    #     X_train = np.array([cv2.imread(filepath) for filepath in X_train])
    #     X_test = np.array([cv2.imread(filepath) for filepath in X_test])
    #
    #     model = Sequential()
    #     # model.add(Conv2D(filters=16, kernel_size=3, input_shape=(height, width, 3), activation="relu"))
    #     # model.add(Dense(16, activation="relu"))
    #     # model.add(Dense(categories.size, activation="softmax"))
    #     model.add(Conv2D(32, (3, 3), padding="same", input_shape=(height, width, 3), activation="relu"))
    #     model.add(MaxPooling2D(pool_size=2))
    #     model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    #     model.add(MaxPooling2D(pool_size=2))
    #     model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    #     model.add(MaxPooling2D(pool_size=2))
    #     model.add(Flatten())
    #     model.add(Dense(64, activation="relu"))
    #     model.add(Dense(6, activation="softmax"))
    #     model.compile(optimizer="adam",
    #                   loss="categorical_crossentropy",
    #                   metrics=["accuracy"])
    #     model.fit(X_train, X_train_labels, batch_size=10, validation_split=0.1, epochs=250)
    #     model.evaluate(X_test, X_test_labels)
    #     model.save("./garbage_classifer.h5")
        # model = Sequential()
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", use_bias=False, input_shape=(height, width, 3)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.1))
        # model.add(Conv2D(filters=64, use_bias=False, kernel_size=(5, 5), strides=2))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        # model.add(Flatten())
        # model.add(Dense(128))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(categories.size, activation="softmax"))
        # model.compile(optimizer="adam",
        #               loss="categorical_crossentropy",
        #               metrics=["accuracy"])
        # X_train = X_train.reshape((len(X_train), num_pixels))
        # model.fit(X_train, X_train_labels, validation_split=0.2, epochs=10, batch_size=64)
        # X_test = X_test.reshape((len(X_test), num_pixels))
        # model.evaluate(X_test, X_test_labels)
        # model.save("garbage_classifier.h5")

    @staticmethod
    def __get_one_hot_encoding(labels, categories):
        rows = len(labels)
        columns = categories.size
        ohe_labels = np.zeros((rows, columns))
        for i in range(len(labels)):
            j = np.where(categories == labels[i])
            ohe_labels[i][j] = 1
        return ohe_labels
