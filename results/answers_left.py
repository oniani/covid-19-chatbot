#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: answers_left.py
Author: David Oniani
E-mail: oniani.david@mayo.edu

Description:
    Generate the file with the answers that are left to be annotated.
"""

import csv

import pandas as pd


def main() -> None:
    """The main function."""

    # Process the data
    answers_all = pd.read_csv("answers_all.csv")
    answers_temp = pd.read_csv("answers_post_annotation.csv")
    answers_post_annotation = pd.DataFrame(
        {
            "Question": answers_temp["Question"],
            "Answer": answers_temp["Answer"],
            "Approach": answers_temp["Approach"],
        }
    )

    rows_all = []
    for _, row in answers_all.iterrows():
        rows_all.append((row[0], row[1], row[2]))

    rows_post_annotation = []
    for _, row in answers_post_annotation.iterrows():
        rows_post_annotation.append((row[0], row[1], row[2]))

    # Filter out the answers
    left = []
    for answer in rows_all:
        if answer not in rows_post_annotation:
            left.append(answer)

    # Write the data to the CSV file
    with open("left_to_annotate.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        # The header
        writer.writerow(
            [
                "Question",
                "Answer",
                "Answer Category - Relevant?",
                "Answer Category - Well-formed?",
                "Answer Category - Informative?",
                "Answer Category - Acceptable?",
                "Answer Category - Poor?",
                "Answer Relevance (1-5)",
                "Approach",
            ]
        )

        for question, answer, approach in left:
            writer.writerow(
                [question, answer, "", "", "", "", "", "", approach,]
            )


if __name__ == "__main__":
    main()
