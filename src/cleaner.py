#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: cleaner.py
Author: David Oniani
E-mail: oniani.david@mayo.edu

Description:
    Clean up the text for further processing.
"""

import re

from typing import List


def clean_text(answers: str) -> List[str]:
    """Clean up the text for further processing.

    NOTE: Usually, `answers` string contains multiple sentences.
    """

    # Chunk the answer into sentences
    answer_list = [
        answer.strip().replace("\n", " ")
        for answer in re.split(r"<|endoftext|>", answers)
        if answer != ""
    ]

    # Eliminate double spaces (if present)
    for idx, item in enumerate(answer_list):
        while "  " in item:
            item = (
                item.replace("  ", " ")
                .replace(" . ", ". ")
                .replace(" , ", ", ")
                .replace(" ? ", "? ")
                .replace(" ! ", "! ")
            )
        answer_list[idx] = item

    # Handle the punctuation
    final_answers = []
    for answer in answer_list:
        if answer.count(".") > 1:
            final_answers.extend(
                [item.strip() + "." for item in answer.split(".")]
            )

        if answer.count("?") > 1:
            final_answers.extend(
                [item.strip() + "?" for item in answer.split("?")]
            )

        if answer.count("!") > 1:
            final_answers.extend(
                [item.strip() + "!" for item in answer.split("!")]
            )

        else:
            final_answers.append(answer)

    return final_answers
