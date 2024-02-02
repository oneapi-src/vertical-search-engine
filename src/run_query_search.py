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
import logging
import os
import pathlib
import pickle
import random
import time
from typing import List, Dict
import yaml

import numpy as np
from sentence_transformers import util
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel
)
from sentence_transformers.cross_encoder import CrossEncoder
from utils.dataloader import load_queries, load_corpus, CorpusDataset
from utils.embed import encode, batch_encode 
random.seed(0)


def search_query(
        logger: logging.Logger,
        tokenizer: PreTrainedTokenizer,
        embedder: PreTrainedModel,
        cross_encoder: CrossEncoder,
        queries: List[str],
        corpus_embeddings: np.ndarray,
        idx_to_ids: Dict[int, str],
        corpus: CorpusDataset,
        top_k: int = 5,
        use_re_ranker: bool=True,
        max_sequence_length: int = 128,
        batch_size: int = 1,
        n_runs: int = 100,
        score: str = 'cos_sim',
        output_file: str = None,
        benchmark_mode: bool = False) -> None:
    """Vector search against a saved set of corpus embeddings.

    Args:
        logger (logging.Logger):
            logger
        tokenizer (PreTrainedTokenizer):
            pretrained tokenizer to to use for tokenizing raw text.
        embedder (PreTrainedModel):
            pretrained model to use to embed documents.
        queries (List[str]):
            List of string queries to search against corpus.
        corpus_embeddings (np.ndarray):
            Pre-embedded corpus of documents.
        idx_to_ids: (Dict[int, int]):
            Map of embedding index to document ids.
        corpus (CorpusDataset):
            CorpusDataset to embed.
        top_k (int, optional):
            Number of entries similar corpus documents to return. 
        max_sequence_length (int, optional):
            max sequence length. Defaults to 128.
        batch_size (int, optional):
            batch size. Defaults to 1.
        n_runs (int, optional):
            number of iterations for benchmarks. Defaults to 100.
        score (str, optional):
            scoring function to use. Defaults to 'cos_sim'.
        output_file (str, optional):
            file to output embeddings to. Defaults to None.
        benchmark_mode (bool, optional):
            whether to run in benchmark mode. Defaults to False.
    """

    # set scoring function
    score_func = util.cos_sim
    if score == 'dot':
        score_func = util.dot_score

    if benchmark_mode:

        times = []

        # sample random inputs for benchmarking execution time
        sample_inputs = [
            random.sample(
                range(tokenizer.vocab_size), max_sequence_length) for
            _ in range(batch_size)]

        sample_inputs = tokenizer.batch_decode(sample_inputs)

        for i in range(10 + n_runs):

            # evaluate query encoding and top-k retrieval time
            start = time.time()

            query_embeddings = encode(
                tokenizer, embedder, sample_inputs,
                max_length=max_sequence_length)

            util.semantic_search(
                query_embeddings,
                corpus_embeddings,
                top_k=top_k,
                score_function=score_func)

            end = time.time()
            if i > 10:
                times.append(end - start)

        logger.info("Batch Size = %d, Max Seq Length = %d, k = %d",
                    batch_size,
                    max_sequence_length,
                    top_k)
        logger.info("Average Inference Time : %f", np.mean(times))

    else:

        # compute query embeddings
        query_embeddings = batch_encode(
            tokenizer,
            embedder,
            queries,
            max_length=max_sequence_length,
            batch_size=batch_size)

        # get nearest corpus item per query top K
        out = util.semantic_search(
            query_embeddings,
            corpus_embeddings,
            top_k=top_k,
            score_function=score_func)
        
        if use_re_ranker and cross_encoder!=None:
            ### Prepare inp for cross_encoder using output of bi_encoder
            for i in range(len(out)):
                cross_inp = [[queries[i], corpus[entry['corpus_id']]] for entry in out[i]]
                cross_scores = cross_encoder.predict(cross_inp)
                for idx in range(len(cross_scores)):
                    out[i][idx]['cross-score'] = float(cross_scores[idx])
                out[i] = sorted(out[i], key=lambda x: x['cross-score'], reverse=True)
            
        # map index based ids to raw corpus_ids
        for i in range(len(out)):
            for entry in out[i]:
                entry['corpus_id'] = idx_to_ids[entry['corpus_id']]

        if output_file is not None:
            path = pathlib.Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as outfile:
                json.dump(out, outfile)
        else:
            print(json.dumps(out, indent=2))


def main(flags):
    """Run embedding of documents using a passed in model.

    Args:
        flags : run flags
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)

    logger = logging.getLogger()

    if not os.path.exists(flags.vse_config):
        logger.error("VSE configuration %s not found!", flags.vse_config)
        return

    # parse the yaml model config
    with open(flags.vse_config, 'r') as stream:
        conf = yaml.safe_load(stream)

    tokenizer = AutoTokenizer.from_pretrained(
        conf['model']['pretrained_model'])

    # load the appropriate model
    if conf["model"]["format"] == "default":

        # load the pretrained embedding model
        embedder = AutoModel.from_pretrained(conf['model']['pretrained_model'])
        cross_encoder = CrossEncoder(conf['model']['cross_encoder'])

    elif conf["model"]["format"] == "inc":

        # load an INC model by loading pretrained model and updating weights
        from neural_compressor.utils.pytorch import load

        embedder = AutoModel.from_pretrained(conf['model']['pretrained_model'])
        embedder = load(conf["model"]["path"], embedder)
        cross_encoder = CrossEncoder(conf['model']['cross_encoder'])
        # re-establish logger because it breaks from above
        logging.getLogger().handlers.clear()

        if flags.logfile == "":
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
        logger = logging.getLogger()

    elif conf["model"]["format"] == "ipex_int8":

        # load a PT saved model saved using torch.save
        if not os.path.exists(conf['model']['path'] + "/saved_model.pt"):
            logger.error("Saved model %s not found!",
                         conf['model']['path'] + "/saved_model.pt")
            return
        embedder = torch.jit.load(conf['model']['path'] + "/saved_model.pt")

    else:
        return

    embedder.eval()
    max_sequence_length = conf['model']['max_seq_length']

    # use IPEX to optimize model

    import intel_extension_for_pytorch as ipex
    embedder = ipex.optimize(embedder, dtype=torch.float32)

    sample_inputs = tokenizer.batch_decode([
        random.sample(
            range(tokenizer.vocab_size), max_sequence_length) for
        _ in range(1)])
    dummy_input = tokenizer(
        sample_inputs,
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors='pt'
    )

    with torch.no_grad():
        embedder = torch.jit.trace(embedder,
                                   [dummy_input['input_ids'],
                                       dummy_input['attention_mask']],
                                   check_trace=False,
                                   strict=False)
        embedder = torch.jit.freeze(embedder)

    # load input queries and dense corpus embeddings
    input_file = load_queries(flags.input_queries)

    with open(conf['inference']['corpus_embeddings_path'], 'rb') as f:
        saved_embeddings = pickle.load(f)
        ids = saved_embeddings['_ids']
        corpus_embeddings = saved_embeddings['embeddings']
        idx_to_ids = dict(enumerate(ids))

    # read in corpus dataset
    corpus = load_corpus(flags.input_corpus)
    search_query(
        logger=logger,
        tokenizer=tokenizer,
        embedder=embedder,
        cross_encoder=cross_encoder,
        queries=input_file.queries,
        corpus_embeddings=corpus_embeddings,
        idx_to_ids=idx_to_ids,
        corpus=corpus,
        top_k=conf['inference']['top_k'],
        max_sequence_length=max_sequence_length,
        batch_size=flags.batch_size,
        use_re_ranker=flags.use_re_ranker,
        n_runs=flags.n_runs,
        score=conf['inference']['score_function'],
        output_file=flags.output_file,
        benchmark_mode=flags.benchmark_mode
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--logfile',
                        type=str,
                        default="",
                        help="Log file to output benchmarking results to.")

    parser.add_argument('--vse_config',
                        type=str,
                        required=True,
                        help="Vertical Search Engine model config yml"
                        )

    parser.add_argument('--input_queries',
                        required=True,
                        help="path to corpus to embed",
                        type=str
                        )

    parser.add_argument('--input_corpus',
                        required=True,
                        help="path to corpus to embed",
                        type=str
                        )

    parser.add_argument('--output_file',
                        required=False,
                        help="file to output top k documents to",
                        type=str,
                        default=None
                        )

    parser.add_argument('--batch_size',
                        required=False,
                        help="batch size for embedding",
                        type=int,
                        default=8
                        )

    parser.add_argument('--benchmark_mode',
                        required=False,
                        help="toggle to benchmark embedding",
                        action="store_true",
                        default=False
                        )

    parser.add_argument('--use_re_ranker',
                        required=False,
                        help="Use cross encoder reranking",
                        action="store_true",
                        default=False
                        )

    parser.add_argument('--n_runs',
                        required=False,
                        help="number of iterations to benchmark embedding",
                        type=int,
                        default=100
                        )

    FLAGS = parser.parse_args()

    main(FLAGS)