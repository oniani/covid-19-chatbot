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

from sklearn import metrics


if __name__ == "__main__":
    annotator_1 = [3, 3, 1, 1, 5, 4, 5, 3, 2, 2, 1, 5, 4, 4, 4, 3, 4, 3, 4, 4]
    annotator_2 = [1, 5, 4, 4, 5, 2, 4, 3, 5, 1, 2, 5, 4, 5, 5, 3, 5, 4, 5, 4]

    print(
        "Cohen's Kappa:", metrics.cohen_kappa_score(annotator_1, annotator_2)
    )
    print(
        "Correlation Coefficient:", np.corrcoef(annotator_1, annotator_2)[0][1]
    )
