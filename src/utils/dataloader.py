# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Helpers for loading datasets
"""
import pandas as pd
from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    """Dataset of corpus entries
    """

    def __init__(self, ids, corpus):
        self.ids = ids
        self.corpus = corpus

    def __getitem__(self, item):
        return self.corpus[item]

    def __len__(self):
        return len(self.corpus)

    def get_corpus(self):
        return self.corpus

    def get_ids(self):
        return self.ids


class QueryDataset(Dataset):
    """Dataset of query entries
    """

    def __init__(self, ids, queries):
        self.ids = ids
        self.queries = queries

    def __getitem__(self, item):
        return self.queries[item]

    def __len__(self):
        return len(self.queries)

    def get_queries(self):
        return self.queries

    def get_ids(self):
        return self.ids


def load_corpus(data_file_path):
    """load the corpus dataset

    Args:
        data_file_path (str): path to data file
    Returns:
        Dataset : Dataset for corpus evaluation
    """
    data = pd.read_csv(data_file_path, index_col=False)
    return CorpusDataset(
        data['_id'].values.tolist(),
        data['corpus'].astype(str).values.tolist())


def load_queries(data_file_path):
    """load the query dataset

    Args:
        data_file_path (str): path to data file
    Returns:
        Dataset : Dataset for query evaluation
    """
    data = pd.read_csv(data_file_path, index_col=False)
    return QueryDataset(
        data['query-id'].values.tolist(),
        data['query'].astype(str).values.tolist())
