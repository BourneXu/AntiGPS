# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 19:46:23
# -----
# Last Modified: 2020-02-18 09:59:07
# Modified By: Chris
# -----
# Copyright (c) 2020
###

import pickle
import random
import concurrent.futures
from typing import List
from collections import defaultdict

import numpy as np
import plyvel
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import spatial
from loguru import logger
from fuzzywuzzy import fuzz
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences

from script.utility import Utility


class Decider:
    def __init__(self):
        logger.info("Initilizing Decider ...")
        pass

    def similarity_text(self, textA, textB):
        ## Fuzzy string matching
        return fuzz.partial_ratio(textA, textB) / 100.0

    def similarity_vector(self, vectorA, vectorB):
        return 1 - spatial.distance.cosine(vectorA, vectorB)

    def init_leveldb(self):
        self.db_attack = plyvel.DB("/home/bourne/Workstation/AntiGPS/results/train_data_attack/")
        self.db_noattack = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_noattack/"
        )
        self.db_attack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_attack_poi/"
        )
        self.db_noattack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/train_data_noattack_poi/"
        )
        self.db_partial_attack = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/test_data_partial_attack/",
        )
        self.db_partial_attack_google = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/test_data_partial_attack_google/",
        )
        self.db_partial_attack_poi = plyvel.DB(
            "/home/bourne/Workstation/AntiGPS/results/test_data_partial_attack_poi/",
        )

    def close_leveldb(self):
        self.db_attack.close()
        self.db_noattack.close()
        self.db_attack_poi.close()
        self.db_noattack_poi.close()

    def load_data_db(self, keys, attack, poi=False):
        if attack:
            if poi:
                return (
                    keys,
                    [pickle.loads(self.db_attack_poi.get(bytes(key))) for key in tqdm(keys)],
                )
            return keys, [pickle.loads(self.db_attack.get(bytes(key))) for key in tqdm(keys)]
        else:
            if poi:
                return (
                    keys,
                    [pickle.loads(self.db_noattack_poi.get(bytes(key))) for key in tqdm(keys)],
                )
            return keys, [pickle.loads(self.db_noattack.get(bytes(key))) for key in tqdm(keys)]

    def load_data(self, test_split=0.2, workers=5, sample=0.5, routes_slot=None, poi=False):
        logger.info("Loading data from local levelDB ...")
        self.init_leveldb()
        data_attack, data_noattack = [], []
        keys_test = []
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
            keys_attack = random.sample(keys_attack, round(len(keys_attack) * sample))
            keys_noattack = random.sample(keys_noattack, round(len(keys_noattack) * sample))
        else:
            keys_attack = [
                str(i).encode()
                for i in random.sample(list(range(routes_slot)), round(routes_slot * sample))
            ]
            keys_noattack = [
                str(i).encode()
                for i in random.sample(
                    list(range(2 * routes_slot, 3 * routes_slot)), round(routes_slot * sample)
                )
            ]
            keys_attack, keys_noattack = (
                np.array_split(keys_attack, workers),
                np.array_split(keys_noattack, workers),
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            r = [
                executor.submit(self.load_data_db, keys_attack[i], True, poi)
                for i in range(workers)
            ]
            for re in concurrent.futures.as_completed(r):
                data_attack.extend(re.result()[1])
                keys_test.extend(re.result()[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            r = [
                executor.submit(self.load_data_db, keys_noattack[i], False, poi)
                for i in range(workers)
            ]
            for re in concurrent.futures.as_completed(r):
                data_noattack.extend(re.result()[1])
        self.close_leveldb()
        logger.debug(f"attack data: {len(data_attack)}, nonattack data: {len(data_noattack)}")
        train_size = int((len(data_attack) + len(data_noattack)) * (1 - test_split))

        num = round(train_size / 2)
        X_train = np.array(data_attack[:num] + data_noattack[:num])
        y_train = np.array([1] * num + [0] * num)
        X_test = np.array(data_attack[num:] + data_noattack[num:])
        y_test = np.array([1] * (len(data_attack) - num) + [0] * (len(data_noattack) - num))

        ## Save keys for testing
        keys_test = keys_test[num:]
        with open("/home/bourne/Workstation/AntiGPS/results/keys_test.pkl", "wb") as fout:
            pickle.dump(keys_test, fout)
        return (pad_sequences(X_train), y_train, pad_sequences(X_test), y_test)

    def create_model(self, input_length, poi=False):
        logger.info("Creating model...")
        model = Sequential()
        if poi:
            model.add(
                LSTM(
                    units=256,
                    activation="sigmoid",
                    recurrent_activation="hard_sigmoid",
                    return_sequences=True,
                    input_shape=(50, 1542),
                )
            )
        else:
            model.add(
                LSTM(
                    units=256,
                    activation="sigmoid",
                    recurrent_activation="hard_sigmoid",
                    return_sequences=True,
                    input_shape=(50, 1548),
                )
            )
        model.add(Dropout(0.5))
        model.add(LSTM(units=256, activation="sigmoid", recurrent_activation="hard_sigmoid"))
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

    def load_test_data_partial_attack(self, rate: float, valid="default"):
        logger.info(f"Loading partial attacked test data with rate: {rate}, valid: {valid}")
        if valid == "default":
            db = self.db_partial_attack
        elif valid == "google":
            db = self.db_partial_attack_google
        elif valid == "poi":
            db = self.db_partial_attack_poi
        else:
            raise ValueError(f"Invalid valid value: {valid}")
        route_keys = self.load_test_keys()
        X_test = []
        for route_key in tqdm(route_keys):
            key = (f"{route_key.decode()}_{rate}").encode()
            X_test.append(pickle.loads(db.get(key)))
        logger.debug(f"Rate: {rate}, {len(X_test)} vectors loaded")
        return np.array(X_test), [1] * len(X_test)

    def load_test_keys(self) -> List[bytes]:
        with open("/home/bourne/Workstation/AntiGPS/results/keys_test.pkl", "rb") as fout:
            route_keys = pickle.load(fout)
        return route_keys


def test_lstm(modelpath=None, poi=False):
    test = Decider()
    X_train, y_train, X_test, y_test = test.load_data(
        sample=0.6, workers=20, routes_slot=5000, poi=poi
    )
    if modelpath:
        logger.info(f"Loading model from {modelpath}")
        model = load_model(modelpath)
    else:
        modelpath = "/home/bourne/Workstation/AntiGPS/results/trained_models/lstm_{}.h5".format(
            poi * "poi"
        )
        try:
            logger.info(f"Loading model from {modelpath}")
            model = load_model(modelpath)
        except:
            model = test.create_model(len(X_train[0]), poi=poi)

            logger.info("Fitting model...")
            hist = model.fit(
                X_train, y_train, batch_size=64, nb_epoch=10, validation_split=0.1, verbose=1
            )
            logger.info("Saving model...")

            model.save(modelpath)
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

    _, ax = plt.subplots()
    _ = Utility.plot_cdf([similarities[idx] for idx in idx_attack])
    _ = Utility.plot_cdf([similarities[idx] for idx in idx_noattack])
    ax.set_title("CDF of vector similarities")
    ax.legend(["Attack", "Non-Attack"])
    ax.xaxis.set_label_text("Similarities")
    ax.yaxis.set_label_text("Percent")
    # plt.show()
    plt.savefig("/home/bourne/Workstation/AntiGPS/results/cdf_similarity_vector.png")


def test_partial_attack_predict(modelpath, rates=[], valid="default"):
    test = Decider()
    test.init_leveldb()
    model = load_model(modelpath)
    acc_all = []
    for rate in rates:
        X_test, y_test = test.load_test_data_partial_attack(rate, valid)
        score, acc = model.evaluate(X_test, y_test, batch_size=1)
        acc_all.append(acc)
        logger.info(f"Rate: {rate}, valid: {valid}")
        logger.info(f"Test score: {score}")
        logger.info(f"Test accuracy: {acc}")
    test.close_leveldb()
    return acc_all


if __name__ == "__main__":
    poi = False
    modelpath = None
    modelpath = "/home/bourne/Workstation/AntiGPS/results/trained_models/lstm_{}.h5".format(
        poi * "poi"
    )
    # test_lstm(modelpath=modelpath, poi=poi)
    # test_similarity_vector()

    rates = [round(x * 0.1, 2) for x in range(0, 1)]
    acc_all = test_partial_attack_predict(modelpath=modelpath, rates=rates, valid="default")
    Utility.plot(rates, acc_all)
    print(rates, acc_all)
