# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Generate vector embeddings for text.
"""

from typing import List, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output, attention_mask):
    return model_output['last_hidden_state'][:, 0]


def encode(
        tokenizer: PreTrainedTokenizer,
        embedder: PreTrainedModel,
        texts: List[str],
        pooling: str = 'cls',
        max_length: int = 128) -> torch.Tensor:
    """Transform raw text to embeddings with pooling using a pretrained
    tokenizer and embedder

    Args:
        tokenizer (PreTrainedTokenizer):
            pretrained tokenizer to use for tokenizing raw text
        embedder (PreTrainedModel):
            pretrained model to use to embed documents
        texts (List[str]):
            list of texts to embed.
        pooling (str, optional):
            type of pooling to do for final output. Defaults to 'mean'.
        max_length (int, optional):
            max length of sequence. Defaults to 128.

    Returns:
        np.ndarray:
            embedded documents as an array
    """
    embedder.eval()
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt')

    # Compute sentence embeddings
    with torch.no_grad():
        model_output = embedder(**encoded_input)

        if pooling == 'cls':
            embeddings = cls_pooling(
                model_output, encoded_input['attention_mask'])
        else:
            embeddings = mean_pooling(
                model_output, encoded_input['attention_mask'])
        embeddings = embeddings.detach()

        return embeddings


def batch_encode(
        tokenizer: PreTrainedTokenizer,
        embedder: PreTrainedModel,
        texts: Union[torch.utils.data.Dataset, List[str]],
        pooling: str = 'cls',
        max_length: int = 128,
        batch_size: int = 32):
    """Encode a set of texts by batch.

    Args:
        tokenizer (PreTrainedTokenizer):
            pretrained tokenizer to use for tokenizing raw text
        embedder (PreTrainedModel):
            pretrained model to use to embed documents
        texts (Union[torch.utils.data.Dataset,List[str]]):
            list of texts to embed.
        max_length (int, optional):
            max length of sequence. Defaults to 128.
        pooling (str, optional):
            type of pooling to do for final output. Defaults to cls.
        batch_size (int, optional):
            batch size to use for encoding. Defaults to 32.

    Returns:
        np.ndarray:
            embedded documents as an array
    """

    dataloader = torch.utils.data.DataLoader(texts, batch_size=batch_size)
    text_embeddings = None

    for _, batch in enumerate(tqdm(dataloader)):
        embeddings = encode(
            tokenizer, embedder, batch, pooling=pooling, max_length=max_length)
        if text_embeddings is not None:
            text_embeddings = torch.cat([text_embeddings, embeddings], 0)
        else:
            text_embeddings = embeddings

    return text_embeddings.numpy()
