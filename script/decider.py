# -*- coding:utf-8 -*-
###
# @Author: Chris
# Created Date: 2020-01-02 19:46:23
# -----
# Last Modified: 2020-01-05 15:42:59
# Modified By: Chris
# -----
# Copyright (c) 2020
###

from fuzzywuzzy import fuzz


class Decider:
    def __init__(self):
        pass

    def similarity_text(self, textA, textB):
        ## Fuzzy string matching
        return fuzz.partial_ratio(textA, textB) / 100.0

    def similarity_position(self):
        pass


if __name__ == "__main__":
    test = Decider()
    prob = test.similarity_text("asdad asd as", "asdasd fesf s")
    print(prob)

