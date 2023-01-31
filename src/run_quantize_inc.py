# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Quantize a model using intel extension for pytorch
"""

import argparse
import os
import logging
import yaml

from neural_compressor.experimental import Quantization, common
import pandas as pd
from sentence_transformers import util
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)

from utils.dataloader import (
    load_queries, load_corpus, QueryDataset, CorpusDataset
)
from utils.embed import batch_encode


def quantize_model(
        tokenizer: PreTrainedTokenizer,
        embedder: PreTrainedModel,
        queries: QueryDataset,
        corpus: CorpusDataset,
        inc_config_file: str,
        score_func,
        gt,
        top_k: int = 5,
        max_seq_length: int = 128,
        batch_size: int = 64):
    """Quantizes the model using the given dataset and INC config

    Args:
        tokenizer (PreTrainedTokenizer):
            pretrained tokenizer to use for tokenizing raw text
        embedder (PreTrainedModel):
            pretrained model to use to embed documents
        queries (QueryDataset):
            dataset of queries to evaluate accuracy.
        corpus (CorpusDataset):
            dataset of corpus to evaluate accuracy.
        inc_config_file (str):
            path to configuration file for INC
        score_func:
            function used to compute the similarity score.
        gt:
            table of ground truth matches for evaluating accuracy.
        top_k (int, optional):
            Number of entries to return for accuracy. Defaults to 5.
        max_seq_length (int, optional):
            max length of sequence. Defaults to 128.
        batch_size (int, optional):
            batch size for evaluation. Defaults to 64.
    """

    query_ids = queries.get_ids()
    corpus_ids = corpus.get_ids()

    def evaluate_contains_top_entries(model_q) -> float:
        query_embeddings = batch_encode(
            tokenizer,
            model_q,
            queries.queries,
            max_length=max_seq_length,
            batch_size=batch_size
        )

        corpus_embeddings = batch_encode(
            tokenizer,
            model_q,
            corpus.corpus,
            max_length=max_seq_length,
            batch_size=batch_size
        )

        res = util.semantic_search(
            query_embeddings,
            corpus_embeddings,
            top_k=top_k,
            score_function=score_func)

        correct = 0
        for idx, query_ranking in enumerate(res):
            matches = []
            for ranking in query_ranking:
                matches.append(corpus_ids[ranking['corpus_id']])

            correct += len(set(gt[gt['query-id'] == query_ids[idx]]
                           ['corpus-id']).intersection(set(matches)))
        return correct/len(gt)

    # quantize model using provided configuration

    tokenized_entries = []
    for x in corpus:
        tokens = tokenizer(x,
                           padding='max_length',
                           max_length=64,
                           truncation=True,
                           return_tensors='pt')
        tokens['attention_mask'] = tokens['attention_mask'][0, :]
        tokens['input_ids'] = tokens['input_ids'][0, :]
        tokenized_entries.append((tokens, 0))

    eval_loader = torch.utils.data.DataLoader(
        tokenized_entries,
        batch_size=batch_size,
        shuffle=True
    )

    quantizer = Quantization(inc_config_file)
    cmodel = common.Model(embedder)
    quantizer.model = cmodel
    quantizer.calib_dataloader = eval_loader
    quantizer.eval_func = evaluate_contains_top_entries
    quantized_model = quantizer()

    return quantized_model


def main(flags) -> None:
    """Calibrate model for int 8 and serialize as a .pt

    Args:
        flags: benchmarking flags
    """

    # Validate Flags
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    if not os.path.exists(flags.inc_config_file):
        logger.error("INC configuration %s not found!", flags.inc_config_file)
        return

    if not os.path.exists(flags.vse_config):
        logger.error("VSE configuration %s not found!", flags.vse_config)
        return

    # Load dataset for quantization
    try:
        corpus_dataset = load_corpus(flags.corpus_file)
        query_dataset = load_queries(flags.query_file)
    except FileNotFoundError as exc:
        logger.error("Please follow instructions to download data.")
        logger.error(exc, exc_info=True)
        return

    # parse the yaml model config
    with open(flags.vse_config, 'r') as stream:
        conf = yaml.safe_load(stream)

    # Compute reference values for comparison to original predictions
    tokenizer = AutoTokenizer.from_pretrained(
        conf['model']['pretrained_model'])
    embedder = AutoModel.from_pretrained(conf['model']['pretrained_model'])
    embedder.eval()

    score_func = util.cos_sim
    if conf['inference']['score_function'] == 'dot':
        score_func = util.dot_score

    ground_truth = pd.read_csv(flags.ground_truth_file)

    # Quantize model using Accuracy Aware Quantization
    quantized_model = quantize_model(
        tokenizer,
        embedder,
        query_dataset,
        corpus_dataset,
        flags.inc_config_file,
        score_func,
        ground_truth,
        top_k=conf['inference']['top_k'],
        max_seq_length=conf["model"]["max_seq_length"],
        batch_size=64)
    quantized_model.save(flags.save_model_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--query_file',
                        required=True,
                        help="query file to use for AAQ",
                        type=str
                        )

    parser.add_argument('--corpus_file',
                        required=True,
                        help="corpus file to use for AAQ",
                        type=str
                        )

    parser.add_argument('--ground_truth_file',
                        required=True,
                        help=" table of query-id and matching corpus-id to evaluate accuracy",
                        type=str
                        )

    parser.add_argument('--vse_config',
                        type=str,
                        required=True,
                        help="Vertical Search Engine model config yml"
                        )

    parser.add_argument('--save_model_dir',
                        type=str,
                        required=True,
                        help="directory to save the quantized model to")

    parser.add_argument('--inc_config_file',
                        help="INC conf yaml",
                        required=True
                        )

    FLAGS = parser.parse_args()

    main(FLAGS)
