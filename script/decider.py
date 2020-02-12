# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 19:46:23
# -----
# Last Modified: 2020-02-11 23:24:42
# Modified By: Chris
# -----
# Copyright (c) 2020
###

import pickle
import concurrent.futures

import numpy as np
import pandas as pd
import plyvel
from tqdm import tqdm
from scipy import spatial
from atpbar import flush, atpbar
from loguru import logger
from fuzzywuzzy import fuzz
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from script.utility import Utility


class Decider:
    def __init__(self):
        pass

    def similarity_text(self, textA, textB):
        ## Fuzzy string matching
        return fuzz.partial_ratio(textA, textB) / 100.0

    def similarity_vector(self, vectorA, vectorB):
        return 1 - spatial.distance.cosine(vectorA, vectorB)

    def init_traindb(self):
        self.db_attack = plyvel.DB("/home/bourne/Workstation/AntiGPS/results/train_data_attack/")
        self.db_noattack = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_noattack/"
        )
        self.db_attack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_attack_poi/",
        )
        self.db_noattack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_noattack_poi/",
        )

    def close_traindb(self):
        self.db_attack.close()
        self.db_noattack.close()
        self.db_attack_poi.close()
        self.db_noattack_poi.close()

    def load_data_db(self, keys, attack, poi=False):
        if attack:
            if poi:
                return [pickle.loads(self.db_attack_poi.get(bytes(key))) for key in atpbar(keys)]
            return [pickle.loads(self.db_attack.get(bytes(key))) for key in atpbar(keys)]
        else:
            if poi:
                return [pickle.loads(self.db_noattack_poi.get(bytes(key))) for key in atpbar(keys)]
            return [pickle.loads(self.db_noattack.get(bytes(key))) for key in atpbar(keys)]

    def load_data(self, test_split=0.2, workers=5, sample=0.5, routes_slot=None, poi=False):
        logger.info("Loading data from local levelDB ...")
        self.init_traindb()
        data_attack, data_noattack = [], []
        if not routes_slot:
            if poi:
                keys_attack = np.array_split(
                    list(self.db_attack_poi.iterator(include_value=False)), workers
                )
                keys_noattack = np.array_split(
                    list(self.db_noattack_poi.iterator(include_value=False)), workers
                )
            else:
                keys_attack = np.array_split(
                    list(self.db_attack.iterator(include_value=False)), workers
                )
                keys_noattack = np.array_split(
                    list(self.db_noattack.iterator(include_value=False)), workers
                )
        else:
            keys_attack = [str(i).encode() for i in range(round(routes_slot * sample))]
            keys_noattack = [
                str(i).encode()
                for i in range(2 * routes_slot, 2 * routes_slot + round(routes_slot * sample))
            ]
            keys_attack, keys_noattack = (
                np.array_split(keys_attack, workers),
                np.array_split(keys_noattack, workers),
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            r = [
                executor.submit(self.load_data_db, keys_attack[i], True, poi)
                for i in range(workers)
            ]
            for re in concurrent.futures.as_completed(r):
                data_attack.extend(re.result())
            flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            r = [
                executor.submit(self.load_data_db, keys_noattack[i], False, poi)
                for i in range(workers)
            ]
            for re in concurrent.futures.as_completed(r):
                data_noattack.extend(re.result())
            flush()
        self.close_traindb()
        logger.debug(f"attack data: {len(data_attack)}, nonattack data: {len(data_noattack)}")
        train_size = int((len(data_attack) + len(data_noattack)) * (1 - test_split))

        num = round(train_size / 2)
        X_train = np.array(data_attack[:num] + data_noattack[:num])
        y_train = np.array([1] * num + [0] * num)
        X_test = np.array(data_attack[num:] + data_noattack[num:])
        y_test = np.array([1] * (len(data_attack) - num) + [0] * (len(data_noattack) - num))

        return (pad_sequences(X_train), y_train, pad_sequences(X_test), y_test)

    def create_model(self, input_length, poi=False):
        logger.info("Creating model...")
        model = Sequential()
        # model.add(Embedding(input_dim=188, output_dim=50, input_length=input_length))
        if poi:
            model.add(
                LSTM(
                    output_dim=256,
                    activation="sigmoid",
                    inner_activation="hard_sigmoid",
                    return_sequences=True,
                    input_shape=(50, 1542),
                )
            )
        else:
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

    def compare_similarity(self, vectors, poi):
        logger.info("Computing vector similarity ...")
        similarities = []
        if poi:
            for vector in tqdm(vectors):
                vectorA = [vector[i][j] for j in range(774) for i in range(50)]
                vectorB = [vector[i][j] for j in range(774, 1542) for i in range(50)]
                similarities.append(self.similarity_vector(vectorA, vectorB))
        else:
            for vector in tqdm(vectors):
                vectorA = [vector[i][j] for j in range(774) for i in range(50)]
                vectorB = [vector[i][j] for j in range(774, 1548) for i in range(50)]
                similarities.append(self.similarity_vector(vectorA, vectorB))
        return similarities


def test_lstm(poi=False):
    test = Decider()
    X_train, y_train, X_test, y_test = test.load_data(
        sample=0.5, workers=10, routes_slot=5000, poi=poi
    )

    model = test.create_model(len(X_train[0]), poi=poi)

    logger.info("Fitting model...")
    hist = model.fit(X_train, y_train, batch_size=64, nb_epoch=10, validation_split=0.1, verbose=1)

    score, acc = model.evaluate(X_test, y_test, batch_size=1)
    logger.info(f"Test score: {score}")
    logger.info(f"Test accuracy: {acc}")


def test_similarity_vector(poi=False):
    test = Decider()
    X_train, y_train, X_test, y_test = test.load_data(
        sample=0.5, workers=10, routes_slot=5000, poi=poi
    )
    similarities = test.compare_similarity(X_train, poi=poi)
    idx_attack = [idx for idx in range(len(y_train)) if y_train[idx] == 1]
    idx_noattack = [idx for idx in range(len(y_train)) if y_train[idx] == 0]

    fig, ax = plt.subplots()
    line1 = Utility.plot_cdf([similarities[idx] for idx in idx_attack])
    line2 = Utility.plot_cdf([similarities[idx] for idx in idx_noattack])
    ax.set_title("CDF of vector similarities")
    ax.legend(["Attack", "Non-Attack"])
    ax.xaxis.set_label_text("Similarities")
    ax.yaxis.set_label_text("Percent")
    # plt.show()
    plt.savefig("/home/bourne/Workstation/AntiGPS/results/cdf_similarity_vector.png")


if __name__ == "__main__":
    # test_lstm(poi=True)
    test_similarity_vector()
