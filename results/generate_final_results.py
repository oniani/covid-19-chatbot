#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: generate_final_results.py
Author: David Oniani
E-mail: oniani.david@mayo.edu

Description:
    Filter answers based on the similarity to the original question.
    This approach uses Universal Sentence Classfier (USE).
"""

import csv
import os
import re
import string

import nltk

import extract

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from typing import List

from bert_serving.client import BertClient

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download("stopwords")
nltk.download("wordnet")


DATA_DIR: str = "answers"


def preprocess(text: str) -> str:
    """Preprocess the sentences."""

    lemmatizer = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words("english")

    text = "".join(
        [word.lower() for word in text if word not in string.punctuation]
    )

    tokens = re.split("\\W+", text)
    result = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stopwords
    ]

    return " ".join(result)


def calculate_similarity(features):
    """Calculate the correlation score.

    The embeddings produced by the USE are approximately normalized. The
    semantic similarity of two sentences can be trivially computed as the inner
    product of the encodings.
    """

    return np.inner(features, features)


def get_features(texts, embed):
    """A simple function to wrap TF call. We create a session and run the embed
       node in the graph. This gives us the vector for each text.
    """

    if type(texts) is str:
        texts = [texts]

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))


def use_filter(
    question: str, answer_list: List[str], num_sentences: int
) -> str:
    """Get a better, filtered answer."""

    # Deal with cases related to the number of sentences
    if len(answer_list) == 0:
        return ""

    elif len(answer_list) == 1:
        return answer_list[0]

    if num_sentences > len(answer_list):
        num_sentences = len(answer_list)

    # Load USE (Universal Sentence Encoder) version 3 - large
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    print("Loading model from {}".format(module_url))
    embed = hub.Module(module_url)

    # Apply the model and calculate the similarity
    #
    # We first make the question the last entry of the answer list. This will
    # make it easy to generate a column with the question-answer correlations
    # for every answer sentence
    #
    # We proceed by creating the embeddings for the `answer_list` and
    # calculating the similarity scores
    answer_list.append(question)
    answers = [preprocess(answer.lower()) for answer in answer_list]
    features = get_features(
        [answer for answer in answers if answer != ""], embed,
    )
    similarity_matrix = calculate_similarity(features)

    # Find `number_of_sentences` number of indices (for the answer sentences)
    # with the highest correlation to the question
    highest = sorted(
        np.argpartition(similarity_matrix[:-1, -1], -num_sentences)[
            -num_sentences:
        ]
    )

    # Prepare the final answer
    #
    # Although the words were converted to lowercase, `answer_list` indices
    # remain the same.
    final_answer = ""
    for idx in highest[:-1]:
        final_answer += answer_list[idx] + " "
    final_answer += answer_list[highest[-1]]

    return final_answer


def tfidfvectorizer_cosine_filter(
    question: str, answer_list: List[str], num_sentences: int
) -> str:
    """Find a cosine similarity between two strings."""

    # Deal with cases related to the number of sentences
    if len(answer_list) == 0:
        return ""

    elif len(answer_list) == 1:
        return answer_list[0]

    if num_sentences > len(answer_list):
        num_sentences = len(answer_list)

    # Find the cosine similarity
    answer_list.append(question)
    answers = [preprocess(answer.lower()) for answer in answer_list]
    vectorized = TfidfVectorizer().fit_transform(
        [answer for answer in answers if answer != ""]
    )
    cosine_similarity_matrix = cosine_similarity(vectorized)

    # Find `number_of_sentences` number of indices (for the answer sentences)
    # with the highest correlation to the question
    highest = sorted(
        np.argpartition(cosine_similarity_matrix[:-1, -1], -num_sentences)[
            -num_sentences:
        ]
    )

    # Prepare the final answer
    #
    # Although the words were converted to lowercase, `answer_list` indices
    # remain the same.
    final_answer = ""
    for idx in highest[:-1]:
        final_answer += answer_list[idx] + " "
    final_answer += answer_list[highest[-1]]

    return final_answer


def bert_cosine_filter(
    question: str, answer_list: List[str], num_sentences: int
) -> str:
    """Find a cosine similarity between two strings using BERT.

    NOTE: First run `bert-serving-start -model_dir uncased_L-12_H-768_A-12`

    NOTE: This also works with BioBERT, however, one must rename files in the
          model directory so that it matches the structure of BERT
    """

    # Deal with cases related to the number of sentences
    if len(answer_list) == 0:
        return ""

    elif len(answer_list) == 1:
        return answer_list[0]

    if num_sentences > len(answer_list):
        num_sentences = len(answer_list)

    # Find the cosine similarity using BERT
    answer_list.append(question)
    bc = BertClient(port=5555, port_out=5556)
    answers = [preprocess(answer.lower()) for answer in answer_list]
    vectorized = bc.encode([answer for answer in answers if answer != ""])
    bert_cosine_similarity_matrix = cosine_similarity(vectorized)

    # Find `number_of_sentences` number of indices (for the answer sentences)
    # with the highest correlation to the question
    highest = sorted(
        np.argpartition(
            bert_cosine_similarity_matrix[:-1, -1], -num_sentences
        )[-num_sentences:]
    )

    # Prepare the final answer
    #
    # Although the words were converted to lowercase, `answer_list` indices
    # remain the same.
    final_answer = ""
    for idx in highest[:-1]:
        final_answer += answer_list[idx] + " "
    final_answer += answer_list[highest[-1]]

    return final_answer


def main() -> None:
    """The main function. Benchmarking is done here."""

    writer = csv.writer(open("use_inner.csv", "w"))
    writer.writerow(["Question", "Answer", "Approach"])

    for filename in os.listdir(DATA_DIR):
        data = [
            extract.chunk_into_sentences(extract.clean_additional(item))
            for item in extract.extract(os.path.join(DATA_DIR, filename))
        ]

        question = data[0][0]
        answer_lists = data[1:]  # All answers (each one is already a list)

        for answer_list in answer_lists:
            writer.writerow(
                [
                    question,
                    use_filter(question, answer_list, 3),
                    "Universal Sentence Encoder Version 3 Large + Inner "
                    "Product (numpy)",
                ]
            )


if __name__ == "__main__":
    main()
