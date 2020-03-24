#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: extract.py
Author:   David Oniani
E-mail:   oniani.david@mayo.edu

Description:
    Extract relevant information from the JSON files.
"""

import os
import json
import shutil

from typing import Any, Dict, List


DIR_READ: str = "data_raw"
DIR_WRITE: str = "data"


def extract(filename: str) -> List[str]:
    """Scraper for getting data from the JSON files.

    Parameter(s):
        filename: Specifies a name of the PDF file to be scraped
    """
    with open(filename, "r") as file:
        data: Dict[str, Any] = json.load(file)

    abstract: List[str] = [data["abstract"][0]["text"]] if data[
        "abstract"
    ] else []
    body_text: List[str] = [item["text"] for item in data["body_text"]]
    abstract.extend(body_text)

    return abstract


def main() -> None:
    """The main function."""

    # Extract the data
    json_data: List[List[str]] = []
    for filename in os.listdir(DIR_READ):
        json_data.append(extract("{}/{}".format(DIR_READ, filename)))

    # Create the directory with files
    if os.path.exists(DIR_WRITE):
        shutil.rmtree(DIR_WRITE)
    os.mkdir(DIR_WRITE)

    for idx, content in enumerate(json_data):  # Populate
        with open("{}/data_{}.txt".format(DIR_WRITE, idx), "w") as file:
            file.writelines("\n".join(content))


if __name__ == "__main__":
    main()
