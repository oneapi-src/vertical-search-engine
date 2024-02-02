# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Run query embedding and return top k for each
"""
import argparse
import json
import pandas as pd


def main(flags):
    """Pretty print rankings.

    Args:
        flags : run flags
    """

    rankings = json.load(open(flags.rankings_file, 'r'))
    queries = pd.read_csv(flags.queries)
    corpus = pd.read_csv(flags.corpus)

    output = []

    for idx, query_ranking in enumerate(rankings):
        new_query = {}
        new_query['query'] = queries.iloc[idx]['query']
        matches = []
        for ranking in query_ranking:
            matches.append(
                corpus[corpus['_id'] == ranking["corpus_id"]]['corpus'].values[0]
            )
        new_query["results"] = matches
        output.append(new_query)
    
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rankings_file',
                        type=str,
                        required=True,
                        help="rankings from query search"
                        )

    parser.add_argument('--queries',
                        required=True,
                        help="raw queries file",
                        type=str
                        )

    parser.add_argument('--corpus',
                        required=True,
                        help="raw corpus file",
                        type=str
                        )

    FLAGS = parser.parse_args()

    main(FLAGS)
