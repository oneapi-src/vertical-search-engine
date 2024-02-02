# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Embed all documents in a provided corpus using the given model.
"""
import argparse
import json
import logging
import pathlib
import pickle
import os
import random
import time
import yaml

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel
)

from utils.dataloader import load_corpus, CorpusDataset
from utils.embed import encode, batch_encode

random.seed(0)


def embed_documents(
        logger: logging.Logger,
        tokenizer: PreTrainedTokenizer,
        embedder: PreTrainedModel,
        corpus: CorpusDataset,
        max_sequence_length: int = 128,
        batch_size: int = 1,
        n_runs: int = 100,
        output_file: str = None,
        benchmark_mode: bool = False) -> None:
    """Embed documents using a pretrained tokenizer and embedder

    Args:
        logger (logging.Logger):
            logger
        tokenizer (PreTrainedTokenizer):
            pretrained tokenizer to to use for tokenizing raw text.
        embedder (PreTrainedModel):
            pretrained model to use to embed documents.
        corpus (CorpusDataset):
            CorpusDataset to embed.
        max_seq_length (int, optional):
            max sequence length. Defaults to 128.
        batch_size (int, optional):
            batch size. Defaults to 1.
        n_runs (int, optional):
            number of iterations for benchmarks. Defaults to 100.
        output_file (str, optional):
            file to output embeddings to. Defaults to None.
        benchmark_mode (bool, optional):
            whether to run in benchmark mode. Defaults to False.
    """

    # Run benchmarks
    if benchmark_mode:

        times = []
        # Generate random sample inputs for benchmarking execution time
        sample_inputs = [
            random.sample(
                range(tokenizer.vocab_size), max_sequence_length) for
            _ in range(batch_size)]

        sample_inputs = tokenizer.batch_decode(sample_inputs)
        for i in range(10 + n_runs):
            start = time.time()
            encode(
                tokenizer, embedder,
                sample_inputs, max_length=max_sequence_length)
            end = time.time()
            if i > 10:
                times.append(end - start)

        logger.info("Batch Size = %d, Max Seq Length = %d",
                    batch_size, max_sequence_length)
        logger.info("Average Inference Time : %f", np.mean(times))

    else:

        start = time.time()

        embeddings = batch_encode(
            tokenizer,
            embedder,
            corpus,
            max_length=max_sequence_length,
            batch_size=batch_size
        )

        end = time.time()

        logger.info("Batch Size = %d, Max Seq Length = %d, Documents = %d",
                    batch_size, max_sequence_length, len(corpus))
        logger.info("Embedding Time : %f", end - start)

        if output_file is not None:
            path = pathlib.Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'wb') as f:
                pickle.dump({
                    '_ids': corpus.get_ids(),
                    'embeddings': embeddings},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            out = []
            for i in range(embeddings.shape[0]):
                out.append({
                    "index": i,
                    "embedding": embeddings[i, :].tolist()
                })
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

    if conf["model"]["format"] == "default":

        # load the pretrained embedding model
        embedder = AutoModel.from_pretrained(conf['model']['pretrained_model'])

    elif conf["model"]["format"] == "inc":

        # load an INC model by loading pretrained model and updating weights
        from neural_compressor.utils.pytorch import load

        embedder = AutoModel.from_pretrained(conf['model']['pretrained_model'])
        embedder = load(conf["model"]["path"], embedder)

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
        embedder = torch.jit.trace(
            embedder,
            [dummy_input['input_ids'], dummy_input['attention_mask']],
            check_trace=False,
            strict=False)
        embedder = torch.jit.freeze(embedder)

    # read in corpus dataset
    corpus = load_corpus(flags.input_corpus)

    embed_documents(
        logger=logger,
        tokenizer=tokenizer,
        embedder=embedder,
        corpus=corpus,
        max_sequence_length=max_sequence_length,
        batch_size=flags.batch_size,
        n_runs=flags.n_runs,
        output_file=flags.output_file,
        benchmark_mode=flags.benchmark_mode)


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

    parser.add_argument('--input_corpus',
                        required=True,
                        help="path to corpus to embed",
                        type=str
                        )

    parser.add_argument('--output_file',
                        required=False,
                        help="file to output corpus embeddings to",
                        type=str,
                        default=None
                        )

    parser.add_argument('--batch_size',
                        required=False,
                        help="batch size for embedding. defaults to 32.",
                        type=int,
                        default=32
                        )

    parser.add_argument('--benchmark_mode',
                        required=False,
                        help="toggle to benchmark embedding",
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
