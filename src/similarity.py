#!/usr/bin/env python3
# encoding: UTF-8

"""
Filename: similarity.py
Date: 2020-03-06 10:36:29 AM
Author: David Oniani
E-mail: oniani.david@mayo.edu

Description:
    Filter answers based on the similarity to the original question.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def calculate_similarity(features):
    """Calculate the correlation score.

    The embeddings produced by the Universal Sentence Encoder are approximately
    normalized. The semantic similarity of two sentences can be trivially
    computed as the inner product of the encodings.
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


def filter_answer(question, answer_list, number_of_sentences):
    """Get a better, filtered answer."""

    # Load USE (Universal Sentence Encoder) version 2
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
    features = get_features(answer_list, embed)
    similarity_matrix = calculate_similarity(features)

    # Find `number_of_sentences` number of indices (for the answer sentences)
    # with the highest correlation to the question
    highest = sorted(
        np.argpartition(-similarity_matrix[:-1, -1], number_of_sentences)[
            :number_of_sentences
        ]
    )

    # Prepare the final answer
    final_answer = ""
    for idx in highest[:-1]:
        final_answer += answer_list[idx] + " "
    final_answer += answer_list[-1]

    return final_answer
