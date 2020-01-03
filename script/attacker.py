# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 21:16:28
# -----
# Last Modified: 2020-01-02 23:23:23
# Modified By: Chris
# -----
# Copyright (c) 2020
###

import json
import random
from tqdm import tqdm
from utility import Utility


class Attacker:
    def __init__(self, datafile):
        self.datafile = datafile
        self.__loaddata()

    def __loaddata(self):
        self.dataset = {}
        with open(self.datafile, "r") as fin:
            for line in tqdm(fin.readlines()):
                data = json.loads(line)
                self.dataset[data["id"]] = data

    def random(self, num):
        pano_attack = random.sample(list(self.dataset.values()), num)
        pano_attack_random = []
        for pano in pano_attack:
            gps_correct = (pano["lat"], pano["lng"])
            gps_attack = Utility.generateGPS_random(1)[0]
            while gps_attack == gps_correct:
                gps_attack = Utility.generateGPS_random(1)[0]
            pano["lat_attack"] = gps_attack[0]
            pano["lng_attack"] = gps_attack[1]
            pano_attack_random.append(pano)
        return pano_attack_random

    def nearby(self, num):
        pass

    def driving(self, num):
        pass


if __name__ == "__main__":
    test = Attacker("../results/pano_text.json")
    test_pano_attack_random = test.random(10)
    print(test_pano_attack_random)
