#!/usr/bin/env python3
# encoding: UTF-8
"""
Filename: interact.py
Date: 2020-03-06 01:48:59 PM
Author: David Oniani
E-mail: oniani.david@mayo.edu

Description:

    This is an implementation of an interactive chatbot that answer questions
    related to Alzheimer's Disease.

    It relies on two state-of-the-art MODELs: GPT-2 and USE (Universal Sentence
    Encoder).

    The code is adapted from the previous work of Dr. Yanshan Wang.
"""

import os
import re
import json

import fire

import numpy as np
import tensorflow as tf

import encoder, model, sample, similarity


# String, which MODEL to use
MODEL_NAME = "774M"

# Integer seed for random number generators, fix seed to reproduce results
SEED = None

# Number of samples to return total
NSAMPLES = 1

# Number of batches (only affects speed/memory). Must divide nsamples
BATCH_SIZE = 1

# Number of tokens in generated text, if None (default), is determined by
# MODEL hyperparameters
LENGTH = None

# Float value controlling randomness in boltzmann distribution. Lower
# TEMPERATURE results in less random completions. As the TEMPERATURE
# approaches zero, the MODEL will become deterministic and repetitive.
# Higher TEMPERATURE results in more random completions.
TEMPERATURE = 1

# Integer value controlling diversity. 1 means only 1 word is considered
# for each step (token), resulting in deterministic completions, while 40
# means 40 words are considered at each step. 0 (default) is a special
# setting meaning no restrictions. 40 generally is a good value.
TOP_K = 40

# Path to parent folder containing MODEL subfolders
# (i.e. contains the <MODEL_NAME> folder)
MODELS_DIR = "models"

# Path to the saved MODEL info
CHECKPOINT = "./checkpoint/run2/model-513"


def main():
    """Run the MODEL interactively."""

    print("\nWelcome to Mayo Clinic's Alzheimer's Disease chatbot!")
    print("The input prompt will appear shortly\n\n")

    models_dir = os.path.expanduser(os.path.expandvars(MODELS_DIR))

    assert NSAMPLES % BATCH_SIZE == 0

    enc = encoder.get_encoder(MODEL_NAME)
    hparams = model.default_hparams()

    with open(os.path.join(models_dir, MODEL_NAME, "hparams.json")) as file:
        hparams.override_from_dict(json.load(file))

    if LENGTH is None:
        length = hparams.n_ctx // 2

    elif LENGTH > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: {}".format(
                hparams.n_ctx
            )
        )

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [BATCH_SIZE, None])
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT)

        while True:
            question = input("AD CHATBOT> ")

            while not question:
                print("Prompt should not be empty!")
                question = input("AD CHATBOT> ")

            context_tokens = enc.encode(question)

            generated = 0
            for _ in range(NSAMPLES // BATCH_SIZE):
                out = sess.run(
                    output,
                    feed_dict={
                        context: [context_tokens for _ in range(BATCH_SIZE)]
                    },
                )[:, len(context_tokens) :]

                # P R O C E S S  T H E  D A T A

                # Build the answers string
                answers = ""  # Filter this out - set to [] every iter
                for idx in range(BATCH_SIZE):
                    generated += 1
                    answers += enc.decode(out[idx])

                # Chunk the answer into sentences
                answer_list = [
                    answer.strip().replace("\n", " ")
                    for answer in re.split(r"<|endoftext|>", answers)
                    if answer != ""
                ]

                # Eliminate double spaces (if present)
                for idx, item in enumerate(answer_list):
                    while "  " in item:
                        item = item.replace("  ", " ")
                    answer_list[idx] = item

                # Handle the punctuation
                final_answers = []
                for idx, answer in enumerate(answer_list):
                    if answer.count(".") > 1:
                        new_items = answer.split(".")
                        temp = [item.strip() + "." for item in new_items[:-1]]
                        temp.append(new_items[-1].strip())
                        final_answers.extend(temp)

                    if answer.count("?") > 1:
                        new_items = answer.split("?")
                        temp = [item.strip() + "?" for item in new_items[:-1]]
                        temp.append(new_items[-1].strip())
                        final_answers.extend(temp)

                    if answer.count("!") > 1:
                        new_items = answer.split("!")
                        temp = [item.strip() + "!" for item in new_items[:-1]]
                        temp.append(new_items[-1].strip())
                        final_answers.extend(temp)

                try:
                    print(similarity.filter_answer(question, final_answers, 5))

                except Exception:
                    print(final_answers)
                    print("Model cannot generate an answer")

            print()
            print("=" * 80)
            print()


if __name__ == "__main__":
    # Suppress (most) logging messages
    import absl
    import logging

    logger = logging.getLogger()
    logger.disabled = True
    absl.logging._warn_preinit_stderr = 0

    # Disable TensorFlow deprecation warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Run
    fire.Fire(main())
