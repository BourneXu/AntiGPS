# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 14:55:25
# -----
# Last Modified: 2020-02-23 19:15:29
# Modified By: Chris
# -----
# Copyright (c) 2020
###

import os
import json

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utility import Utility


class Analysis:
    def __init__(self):
        pass

    def load_panotext_json(self, filename):
        self.pano_analysis = pd.DataFrame()
        with open(filename, "r") as fin:
            for line in tqdm(fin.readlines()):
                pano_text = json.loads(line)
                num_total = len(pano_text["text_ocr"])
                num_95 = 0
                for text_info in pano_text["text_ocr"]:
                    if text_info["confidence_score"] >= 0.95:
                        num_95 += 1
                info = {"id": pano_text["id"], "total": num_total, "g95": num_95}
                self.pano_analysis = self.pano_analysis.append(info, ignore_index=True)

    def plot_text_cdf(self, filename):
        _, ax = plt.subplots()
        _ = Utility.plot_cdf(self.pano_analysis["total"])
        _ = Utility.plot_cdf(self.pano_analysis["g95"])
        ax.set_title("CDF of text boxes number")
        ax.legend(["total", "g95"])
        ax.xaxis.set_label_text("Number of text boxes")
        ax.yaxis.set_label_text("Percent")
        # plt.show()
        plt.savefig(filename)


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    test = Analysis()
    test.load_panotext_json(f"{path}/../results/pano_text.json")
    # print(test.pano_analysis)
    test.plot_text_cdf(f"{path}/../results/cdf_pano_text.png")
