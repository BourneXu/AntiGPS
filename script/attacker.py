# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 21:16:28
# -----
# Last Modified: 2020-01-17 13:42:21
# Modified By: Chris
# -----
# Copyright (c) 2020
###

import json
import random
from tqdm import tqdm
from script.utility import Utility


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

    def random_same(self, num):
        return random.sample(list(self.dataset.values()), num)

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

    def driving(self, num_route, num_point):
        pano_init = random.sample(list(self.dataset.values()), num_route)
        pano_attack_driving = []
        for pano in pano_init:
            pano_attack_driving.append(self.generate_route(pano, num_point))
        return pano_attack_driving

    def driving_same(self, num_route, num_point):
        pano_init = random.sample(list(self.dataset.values()), num_route)
        pano_attack_driving_same = []
        for pano in pano_init:
            pano_attack_driving_same.append(self.generate_route(pano, num_point, attack=False))
        return pano_attack_driving_same

    def generate_route(self, pano_init, num_point, attack=True):
        route = []
        pano = pano_init
        for idx in range(num_point):
            if not route:
                pano_nxt_id = pano["neighbor"][0]
            else:
                if pano["neighbor"][0] == route[idx - 1]["id"]:
                    pano_nxt_id = pano["neighbor"][1]
                else:
                    pano_nxt_id = pano["neighbor"][0]
            if attack:
                ## fraud GPS
                gps_correct = (pano["lat"], pano["lng"])
                gps_attack = Utility.generateGPS_random(1)[0]
                while gps_attack == gps_correct:
                    gps_attack = Utility.generateGPS_random(1)[0]
                pano["lat_attack"] = gps_attack[0]
                pano["lng_attack"] = gps_attack[1]
            route.append(pano)
            pano = self.dataset[pano_nxt_id]
        return route


def test_generate_route():
    test = Attacker("../results/pano_text.json")
    routes = test.driving(10, 50)
    coords = {"lats": [], "lngs": []}
    for route in routes:
        for pano in route:
            coords["lats"].append(pano["lat"])
            coords["lngs"].append(pano["lng"])
    Utility.visualize_map(coords)


if __name__ == "__main__":
    # test = Attacker("../results/pano_text.json")
    # test_pano_attack_random = test.random(10)
    # print(test_pano_attack_random)

    test_generate_route()
