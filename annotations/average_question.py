#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: average_question.py
Author:   David Oniani
E-mail:   oniani.david@mayo.edu

Description:
    Report average metrics.
"""

import numpy as np
import pandas as pd


def avg_question(df: pd.DataFrame, name: str) -> float:
    return round(
        (df.loc[df["Question"] == name]["Answer Relevance (1-5)"]).mean(), 3
    )


if __name__ == "__main__":
    annotator_1 = pd.read_csv("annotator_1.csv")
    annotator_2 = pd.read_csv("annotator_2.csv")

    print("Annotator 1")
    print("===========")

    print(
        "Are there geographic variations in the mortality rate of COVID-19?",
        avg_question(
            annotator_1,
            "Are there geographic variations in the mortality rate of "
            "COVID-19?",
        ),
    )

    print(
        "What is known about transmission, incubation, and environmental "
        "stability of COVID-19?",
        avg_question(
            annotator_1,
            "What is known about transmission, incubation, and environmental "
            "stability of COVID-19?",
        ),
    )

    print(
        "Is there any evidence to suggest geographic based virus mutations of "
        "COVID-19?",
        avg_question(
            annotator_1,
            "Is there any evidence to suggest geographic based virus "
            "mutations of COVID-19?",
        ),
    )

    print(
        "Are there geographic variations in the rate of COVID-19 spread?",
        avg_question(
            annotator_1,
            "Are there geographic variations in the rate of COVID-19 spread?",
        ),
    )

    print(
        "What do we know about virus genetics, origin, and evolution of "
        "COVID-19?",
        avg_question(
            annotator_1,
            "What do we know about virus genetics, origin, and evolution of "
            "COVID-19?",
        ),
    )

    print(
        "What has been published about ethical and social science "
        "considerations of COVID-19?",
        avg_question(
            annotator_1,
            "What has been published about ethical and social science "
            "considerations of COVID-19?",
        ),
    )

    print(
        "What has been published about medical care of COVID-19?",
        avg_question(
            annotator_1,
            "What has been published about medical care of COVID-19?",
        ),
    )

    print(
        "What do we know about diagnostics and surveillance of COVID-19?",
        avg_question(
            annotator_1,
            "What do we know about diagnostics and surveillance of COVID-19?",
        ),
    )

    print(
        "What do we know about COVID-19 risk factors?",
        avg_question(
            annotator_1, "What do we know about COVID-19 risk factors?",
        ),
    )

    print(
        "What has been published about information sharing and inter-sectoral "
        "collaboration of COVID-19?",
        avg_question(
            annotator_1,
            "What has been published about information sharing and "
            "inter-sectoral collaboration of COVID-19?",
        ),
    )

    print(
        "What do we know about vaccines and therapeutics of COVID-19?",
        avg_question(
            annotator_1,
            "What do we know about vaccines and therapeutics of COVID-19?",
        ),
    )

    print(
        "What do we know about non-pharmaceutical interventions of COVID-19?",
        avg_question(
            annotator_1,
            "What do we know about non-pharmaceutical interventions of "
            "COVID-19?",
        ),
    )

    ###########################################################################

    print("\n")

    ###########################################################################

    print("Annotator 2")
    print("===========")

    print(
        "Are there geographic variations in the mortality rate of COVID-19?",
        avg_question(
            annotator_2,
            "Are there geographic variations in the mortality rate of "
            "COVID-19?",
        ),
    )

    print(
        "What is known about transmission, incubation, and environmental "
        "stability of COVID-19?",
        avg_question(
            annotator_2,
            "What is known about transmission, incubation, and environmental "
            "stability of COVID-19?",
        ),
    )

    print(
        "Is there any evidence to suggest geographic based virus mutations of "
        "COVID-19?",
        avg_question(
            annotator_2,
            "Is there any evidence to suggest geographic based virus "
            "mutations of COVID-19?",
        ),
    )

    print(
        "Are there geographic variations in the rate of COVID-19 spread?",
        avg_question(
            annotator_2,
            "Are there geographic variations in the rate of COVID-19 spread?",
        ),
    )

    print(
        "What do we know about virus genetics, origin, and evolution of "
        "COVID-19?",
        avg_question(
            annotator_2,
            "What do we know about virus genetics, origin, and evolution of "
            "COVID-19?",
        ),
    )

    print(
        "What has been published about ethical and social science "
        "considerations of COVID-19?",
        avg_question(
            annotator_2,
            "What has been published about ethical and social science "
            "considerations of COVID-19?",
        ),
    )

    print(
        "What has been published about medical care of COVID-19?",
        avg_question(
            annotator_2,
            "What has been published about medical care of COVID-19?",
        ),
    )

    print(
        "What do we know about diagnostics and surveillance of COVID-19?",
        avg_question(
            annotator_2,
            "What do we know about diagnostics and surveillance of COVID-19?",
        ),
    )

    print(
        "What do we know about COVID-19 risk factors?",
        avg_question(
            annotator_2, "What do we know about COVID-19 risk factors?",
        ),
    )

    print(
        "What has been published about information sharing and inter-sectoral "
        "collaboration of COVID-19?",
        avg_question(
            annotator_2,
            "What has been published about information sharing and "
            "inter-sectoral collaboration of COVID-19?",
        ),
    )

    print(
        "What do we know about vaccines and therapeutics of COVID-19?",
        avg_question(
            annotator_2,
            "What do we know about vaccines and therapeutics of COVID-19?",
        ),
    )

    print(
        "What do we know about non-pharmaceutical interventions of COVID-19?",
        avg_question(
            annotator_2,
            "What do we know about non-pharmaceutical interventions of "
            "COVID-19?",
        ),
    )
