# TODO: feature engineering
import bisect
import operator
from collections import defaultdict

import numpy as np
from loguru import logger
from sklearn.cluster import SpectralClustering

from sentence_transformers import SentenceTransformer


class Feature:
    def __init__(self):
        logger.info("Initializing sentence transformer model and text boxes cluster")
        self.sentence_model = SentenceTransformer("bert-base-nli-mean-tokens")
        self.text_box_cluster = SpectralClustering(
            n_clusters=3, assign_labels="discretize", random_state=0
        )

    def textbox_position(self, ocr_results, height=None, width=None, area_num=6):
        if not height or not width:
            height, width = ocr_results["height"], ocr_results["width"]
        text_ocr = ocr_results.get("text", []) or ocr_results.get("text_ocr", [])
        postions = [0] * area_num
        demarcation = np.linspace(0, width, area_num + 1)
        for text_info in text_ocr:
            location = text_info["location"]
            center_coord_x = (location[2] + location[3]) / 2
            if bisect.bisect(demarcation, center_coord_x) > area_num:
                postions[-1] += 1
            else:
                postions[bisect.bisect(demarcation, center_coord_x) - 1] += 1
        return postions

    def object_recognition(self):
        ## recognize like signs and extract texts
        pass

    def traffic_sign_detection(self):
        ## traffic sign detection
        pass

    def sentence_vector(self, ocr_results):
        text_ocr = ocr_results.get("text", []) or ocr_results.get("text_ocr", [])
        text_ocr = sorted(text_ocr, key=lambda x: (x["location"][2], x["location"][0]))
        locations = np.array([x["location"] for x in text_ocr])
        text_groups = defaultdict(str)
        group_order = []
        sentence = ""
        if len(locations) < 2:
            for text in text_ocr:
                if text["confidence_score"] > 0.6:
                    sentence += text["predicted_labels"]
        else:
            ## cluster text boxes
            clustering = self.text_box_cluster.fit(locations)
            for idx, group in enumerate(clustering.labels_):
                if group not in group_order:
                    group_order.append(group)
                text, confidence_socre = (
                    text_ocr[idx]["predicted_labels"],
                    text_ocr[idx]["confidence_score"],
                )
                if confidence_socre > 0.6:
                    text_groups[group] += " " + text.strip(" ")
            for group in group_order:
                sentence += " " + text_groups[group].strip(" ")
        sentence_embeddings = self.sentence_model.encode([sentence.strip(" ")])
        return sentence_embeddings[0]

    def sent2vec(self, sentences):
        return self.sentence_model.encode(sentences)


def test_sentence_vector():
    pass


if __name__ == "__main__":
    f = Feature()
    ocr_results = {
        "height": 600,
        "text": [
            {
                "confidence_score": 0.9239890575408936,
                "location": [189, 248, 1210, 1328],
                "predicted_labels": "jen",
            },
            {
                "confidence_score": 0.14774495363235474,
                "location": [209, 255, 1355, 1441],
                "predicted_labels": "lly",
            },
            {
                "confidence_score": 0.9885714054107666,
                "location": [225, 279, 1442, 1586],
                "predicted_labels": "exchange",
            },
            {
                "confidence_score": 0.9843267798423767,
                "location": [231, 268, 1126, 1192],
                "predicted_labels": "ges",
            },
            {
                "confidence_score": 0.6390743255615234,
                "location": [273, 298, 621, 688],
                "predicted_labels": "alley",
            },
            {
                "confidence_score": 0.6328596472740173,
                "location": [277, 305, 688, 782],
                "predicted_labels": "nationalbank",
            },
        ],
        "width": 1920,
    }
