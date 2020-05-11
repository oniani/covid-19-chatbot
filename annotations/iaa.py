#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: iaa.py
Author:   David Oniani
E-mail:   oniani.david@mayo.edu

Description:
    Reports metrics for Inner Annotation Agreement (IAA).
"""

import numpy as np
import pandas as pd

from sklearn import metrics


if __name__ == "__main__":
    annotator_1 = pd.read_csv("annotator_1.csv")["Answer Relevance (1-5)"]
    annotator_2 = pd.read_csv("annotator_2.csv")["Answer Relevance (1-5)"]

    print(
        "Cohen's Kappa:", metrics.cohen_kappa_score(annotator_1, annotator_2)
    )
    print(
        "Correlation Coefficient:", np.corrcoef(annotator_1, annotator_2)[0][1]
    )
