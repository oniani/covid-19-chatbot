#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: extract.py
Author: David Oniani
E-mail: oniani.david@mayo.edu

Description:
    Extract and clean up the data.
"""

import re

from typing import List


def extract(filename: str) -> List[str]:
    """Extract the data."""
    data = [line.strip() for line in open(filename).readlines()]
    data = [line for line in data if line != 79 * "-" and line != ""]

    chunks = []
    indices = []
    for idx, line in enumerate(data):
        if "| QUESTION" in line:
            if "| ANSWER #" in data[idx + 3]:
                chunks.append(f"{data[idx+1]} {data[idx+2]}")
            else:
                chunks.append(data[idx + 1])

        if "| ANSWER" in line:
            indices.append(idx)

    indices.append(len(data))
    for start_idx, end_idx in zip(indices, indices[1:]):
        chunks.append(" ".join(data[start_idx + 1 : end_idx]))

    return chunks


def clean_additional(string: str) -> str:
    """Additional cleanup."""

    clean_string = re.sub(r"(?:\d+,\s*)+\d+", "", string)
    clean_string = re.sub(r"\[\d*\]", "", clean_string)

    for _ in range(2):
        clean_string = (
            clean_string.replace("  ", " ")
            .replace(" . ", ". ")
            .replace(" , ", ", ")
            .replace(" ? ", "? ")
            .replace(" ! ", "! ")
        )

    return clean_string


def chunk_into_sentences(string: str) -> List[str]:
    """Chunks a long string into a list of sentences."""

    return re.split(r"\.\s", string)
