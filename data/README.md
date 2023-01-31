# Dataset Preparation

The dataset used for this is a question-answering dataset called HotpotQA[1] with a corpus of short answers and a set of queries built on top of the corpus.  The original source of the dataset can be found at https://hotpotqa.github.io/.

In this reference kit, for computational time and demonstration purposes, we will be truncating the dataset from ~2M entries to 200k.  To prepare the dataset for use, you can use the provided `download_data.py` script which downloads a pre-compiled version of this dataset using the `dataset` package provided by HuggingFace and distributed by the Benchmarking IR project (https://github.com/beir-cellar/beir [2]).

The following command downloads the full dataset and creates a truncated version of the original corpus for embedding, and another truncated version for quantization evaluation.

```shell
conda activate vse_stock
python download_data.py
```

> **Please see this data set's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.**
> **HotpotQA is distributed under a [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).**

[1] Yang, Zhilin, et al. "HotpotQA: A dataset for diverse, explainable multi-hop question answering." arXiv preprint arXiv:1809.09600 (2018).

[2] Thakur, Nandan, et al. "BEIR: A heterogenous benchmark for zero-shot evaluation of information retrieval models." arXiv preprint arXiv:2104.08663 (2021).
