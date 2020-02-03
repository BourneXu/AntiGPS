# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 19:46:23
# -----
# Last Modified: 2020-02-02 14:56:57
# Modified By: Chris
# -----
# Copyright (c) 2020
###

import pickle

import numpy as np
import pandas as pd
from loguru import logger
from fuzzywuzzy import fuzz

from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


class Decider:
    def __init__(self):
        pass

    def similarity_text(self, textA, textB):
        ## Fuzzy string matching
        return fuzz.partial_ratio(textA, textB) / 100.0

    def load_data(self, test_split=0.2):
        logger.info("Loading data...")
        with open("/home/bourne/Workstation/AntiGPS/results/data_attack_lstm.pkl", "rb") as fin:
            data_attack = pickle.load(fin)
        with open("/home/bourne/Workstation/AntiGPS/results/data_noattack_lstm.pkl", "rb") as fin:
            data_noattack = pickle.load(fin)

        train_size = int((len(data_attack) + len(data_noattack)) * (1 - test_split))

        num = round(train_size / 2)
        X_train = np.array(data_attack[:num] + data_noattack[:num])
        y_train = np.array([1] * num + [0] * num)
        X_test = np.array(data_attack[num:] + data_noattack[num:])
        y_test = np.array([1] * (len(data_attack) - num) + [0] * (len(data_noattack) - num))

        return (pad_sequences(X_train), y_train, pad_sequences(X_test), y_test)

    def create_model(self, input_length):
        logger.info("Creating model...")
        model = Sequential()
        # model.add(Embedding(input_dim=188, output_dim=50, input_length=input_length))
        model.add(
            LSTM(
                output_dim=256,
                activation="sigmoid",
                inner_activation="hard_sigmoid",
                return_sequences=True,
                input_shape=(50, 1548),
            )
        )
        model.add(Dropout(0.5))
        model.add(LSTM(output_dim=256, activation="sigmoid", inner_activation="hard_sigmoid"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        logger.info("Compiling...")
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        return model


def test_lstm():
    test = Decider()
    X_train, y_train, X_test, y_test = test.load_data()

    model = test.create_model(len(X_train[0]))

    logger.info("Fitting model...")
    hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=10, validation_split=0.1, verbose=1)

    score, acc = model.evaluate(X_test, y_test, batch_size=1)
    logger.info(f"Test score: {score}")
    logger.info(f"Test accuracy: {acc}")


if __name__ == "__main__":
    test_lstm()
