# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Download and setup dataset for benchmarking
"""

from datasets import load_dataset
import pandas as pd

# download the corpus
corpus = load_dataset("BeIR/hotpotqa", "corpus")['corpus'].to_pandas()
corpus.columns = ["_id", "title", "corpus"]

# download the queries
queries = load_dataset("BeIR/hotpotqa", "queries")['queries'].to_pandas()

# download the query-corpus mapping
qrel = load_dataset("BeIR/hotpotqa-qrels")

# turn the queries into text
df = pd.merge(
    qrel['test'].to_pandas(),
    queries,
    left_on="query-id",
    right_on="_id",
    how="left")[["query-id", "text"]]
df.columns = ["query-id", "query"]

# save the test queries
df[["query-id", "query"]].drop_duplicates().to_csv("test_queries.csv", index=False)

# create an abbreviated version of the corpus containing all of the entries in the test set + some random entries
corpus_abbreviated = corpus[corpus['_id'].astype(int).isin(
    qrel['test'].to_pandas()['corpus-id'].values)]
corpus_abbreviated = pd.concat([
    corpus_abbreviated,
    corpus[
        ~corpus['_id'].astype(int)
        .isin(qrel['test'].to_pandas()['corpus-id'].values)]
    .sample(n=200000 - len(corpus_abbreviated), random_state=42)])
corpus_abbreviated[["_id", "corpus"]].to_csv(
    "corpus_abbreviated.csv", index=False)

# take a smaller subset of queries for quantization evaluation to save time
df[["query-id", "query"]].head(1000).drop_duplicates().to_csv(
    "quant_queries.csv", index=False)
corpus_quantization = corpus[corpus['_id'].astype(int).isin(
    qrel['test'].to_pandas().head(1000)['corpus-id'].values)]
corpus_quantization = pd.concat([
    corpus_quantization,
    corpus[
        ~corpus['_id'].astype(int)
        .isin(qrel['test'].to_pandas().head(1000)['corpus-id'].values)]
    .sample(n=2000 - len(corpus_quantization), random_state=42)])
corpus_quantization[["_id", "corpus"]].to_csv(
    "corpus_quantization.csv", index=False)

# save the ground truth for evaluation
qrel['test'].to_pandas().head(1000).drop_duplicates().to_csv(
    "ground_truth_quant.csv", index=False)
