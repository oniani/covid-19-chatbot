#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: average_approach.py
Author:   David Oniani
E-mail:   oniani.david@mayo.edu

Description:
    Report average metrics.
"""

import numpy as np
import pandas as pd


def avg_approach(df: pd.DataFrame, name: str) -> float:
    return round(
        (df.loc[df["Approach"] == name]["Answer Relevance (1-5)"]).mean(), 3
    )


if __name__ == "__main__":
    annotator_1 = pd.read_csv("annotator_1.csv")
    annotator_2 = pd.read_csv("annotator_2.csv")

    print("Annotator 1")
    print("===========")

    print(
        "TfidfVectorizer + Cosine:",
        avg_approach(
            annotator_1,
            "TfidfVectorizer (Scikit-learn) + Cosine Similarity "
            "(Scikit-learn)",
        ),
    )

    print(
        "BERT:",
        avg_approach(
            annotator_1,
            "BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340M "
            "parameters + Cosine Similarity (Scikit-learn)",
        ),
    )

    print(
        "BioBERT:",
        avg_approach(
            annotator_1,
            "BioBERT-Large v1.1 (+ PubMed 1M) - based on BERT-large-Cased "
            "(custom 30k vocabulary) + Cosine Similarity (Scikit-learn)",
        ),
    )

    print(
        "USE:",
        avg_approach(
            annotator_1,
            "Universal Sentence Encoder Version 3 Large + Inner Product "
            "(numpy)",
        ),
    )

    ###########################################################################

    print("\n")

    ###########################################################################

    print("Annotator 2")
    print("===========")

    print(
        "TfidfVectorizer + Cosine:",
        avg_approach(
            annotator_2,
            "TfidfVectorizer (Scikit-learn) + Cosine Similarity "
            "(Scikit-learn)",
        ),
    )

    print(
        "BERT:",
        avg_approach(
            annotator_2,
            "BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340M "
            "parameters + Cosine Similarity (Scikit-learn)",
        ),
    )

    print(
        "BioBERT:",
        avg_approach(
            annotator_2,
            "BioBERT-Large v1.1 (+ PubMed 1M) - based on BERT-large-Cased "
            "(custom 30k vocabulary) + Cosine Similarity (Scikit-learn)",
        ),
    )

    print(
        "USE:",
        avg_approach(
            annotator_2,
            "Universal Sentence Encoder Version 3 Large + Inner Product "
            "(numpy)",
        ),
    )
