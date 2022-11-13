# De-Ranker
This repository contains the code and resources for our paper:
- [Dealing with Textual Noise for Robust and Effective BERT Re-ranking](https://www.sciencedirect.com/science/article/pii/S0306457322002369). 
In Information Processing & Management, Volume 60, Issue 1, 2023.

![image](https://github.com/cxa-unique/De-Ranker/blob/main/deranker_framework.png)

## Textual Noise Simulation
In IR community, there is a lack of available parallel clean and noisy dataset to support
the noise-robustness investigation on BERT re-ranker. Meanwhile, it is infeasible to 
clean unstructured raw text within millions of documents in a noisy dataset, and
a more common approach is to inject synthetic noise into a relatively clean dataset.

Thus, to carry out a quantitative study, we choose to simulate different *within-document* 
textual noises, including `NrSentH`, `NrSentM`, `NrSentT`, `DupSent`, `RevSent`, `NoSpace`, 
`RepSyns`, `ExtraPunc`, `NoPunc`, `MisSpell`. You can find more details of these textual noises 
in our paper (in Section 2.2). Herein, you can use `simulate_textual_noise.py` script to insert 
synthetic textual noise into top candidate documents or into the whole corpus. Before using it, 
you may need to install [TextFlint](https://github.com/textflint/textflint), 
[spaCy](https://spacy.io/usage) and [NLTK](https://www.nltk.org) toolkits. 
```
# insert one specific noise into top candidates
python simulate_textual_noise.py --simulate_mode 'top'
                                 --qrels_file # annotations
                                 --top_file # top candidate file
                                 --query_file # query file 
                                 --output_pairs_file # output noisy top candidate file
                                 --noise_type 'NrSentH' or others

# insert all types of nosie into original corpus
python simulate_textual_noise.py --simulate_mode 'corpus'
                                 --corpus_file # corpus file 
                                 --output_corpus_file # output noisy corpus file
```
Besides, we have provided our generated synthetic noisy data on top of MS MARCO dataset
in [Resources](https://github.com/cxa-unique/De-Ranker/#Resources) for future research,
including the noisy version of MS MARCO (`MS MARCO w/ Noise`), and all noisy initial ranking 
lists on three test sets.

## Model Training
Before training, you need to create a new environment `conda create -n deranker python=3.7`, and install 
a few basic packages, such as `torch==1.3.0`, `tensorflow==1.14.0` and `apex==0.1`.

As for the training data, we use `sample_train_qidpidtriples.py` script to sample a set of training triples 
from the official file 
[qidpidtriples.train.full.2.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz) 
to construct our original training data `D_O`, and we also need to convert training samples into features 
using `convert_triples_to_features.py` script before feeding them into the model.
For reproducibility, we release our used train triple ids `qidpidtriples.train.full.2.sampled.p1n10.ids.tsv` 
in [Resources](https://github.com/cxa-unique/De-Ranker/#Resources).

**BERT_O / BERT_O+N:** This is vanilla BERT re-ranker, using a cross-entropy loss to fine-tune BERT model
with a two-class classification layer. BERT_O is only trained on original training data `D_O`.
As for simple noise augmentation, we replace the original text in `D_O` with the corresponding
noisy version in `MS MARCO w/ Noise` to construct the noisy training data `D_N`. Then, we can add
noisy training samples in `D_N` into `D_O` to obtain `D_O+N`, and BERT_O+N is trained on this `D_O+N`.
For both BERT_O and BERT_O+N, you can refer to `vanilla_bert_finetune.sh` for the model training.

**De-Ranker:** This is our proposed noise-tolerant BERT re-ranker, by learning a noise-invariant relevance prediction 
or representation. We design two versions of De-Ranker using two kinds of denoising methods, namely, Dynamic-Denoising 
and Static-Denoising, according to whether the supervision signal from original text is changed or not during training.
Similarly, we can insert the noisy version of original text in `MS MARCO w/ Noise` into `D_O` to obtain a parallel 
training data `D_O-N`. For both De-Ranker_DD and De-Ranker_SD, you can refer to `deranker_finetune_dd.sh` and 
`deranker_finetune_sd.sh` for the model training, respectively.


## Evaluation
We provide `run_bert_rerank_eval.sh` script to perform re-ranking on initial ranking lists returned by first-stage 
retrieval models, such as BM25. Before that, we need to convert evaluation query-passage pairs into features using 
`convert_pairs_to_features.py` script. We have released our used synthetic initial ranking lists 
in [Resources](https://github.com/cxa-unique/De-Ranker/#Resources).
Each text set (Dev, TREC 2019-2020 DL) contains one original initial ranking list with relatively clean text, 
and ten types of noisy initial ranking lists. By comparing re-ranking results of BERT_O on these initial ranking lists, 
we can examine the individual impact of different synthetic textual noises.

In our experiments, we also investigate whether these BERT re-rankers, especially our proposed De-Ranker, can
effectively tackle natural textual noise in real-world text. That is, we further use 4 widely-used IR datasets 
([TREC CAR](https://trec-car.cs.unh.edu/datareleases/v2.0-release.html), 
[ClueWeb09-B](https://lemurproject.org/clueweb09), 
[Gov2](http://ir.dcs.gla.ac.uk/test_collections/gov2-summary.htm) and
[Robust04](https://trec.nist.gov/data/t13_robust.html))
for zero-shot robustness testing, wherein **ClueWeb09-B** and **Gov2** datasets contain lots of textual noise. 
After preparing the data, you can produce initial ranking list in the format of `q_id \t p_id \t q_text \t p_text \n`,
and use `convert_pairs_to_features.py` and `run_bert_rerank_eval.sh` scripts for this evaluation.

Besides, we use other 14 publicly available datasets in the [BEIR](https://github.com/beir-cellar/beir) benchmark
to examine the zero-shot domain transfer ability of these BERT re-rankers, which is more in line with practical 
applications. Herein, we provide `run_beir_retrieve_rerank.sh` script for both retrieval and re-ranking on the BEIR 
benchmark. You may need to download and turn on [Elasticsearch](https://www.elastic.co/cn/downloads/elasticsearch), 
and use an another environment with
 `sentence-transformers==2.2.0`, `transformers==4.18.0`, `torch==1.11.0` and `beir==1.0.0`.

## Resources
### Synthetic Noisy Data:

1. Noisy version of MS MARCO: [MS MARCO (w/ Noise)](https://drive.google.com/file/d/1nP_ssjGF3g9s_pVLhr5-4FzjiyEw2K9v/view?usp=sharing)
    - It is parallel to original MS MARCO passage corpus: [collection.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz)
    - Format: `p_id \t p_text \t noise_type \n`

2. Noisy initial ranking list:

    | MS MARCO Dev | TREC 2019 DL | TREC 2020 DL 
    |-----|-----|-----|
    | docT5query top-100 | BM25 top-1000 | BM25 top-1000 |
    | [Download](https://drive.google.com/drive/folders/1WDJvrwX2AdDj3njtY6dgsbIIsDGftxBr?usp=sharing) | [Download](https://drive.google.com/drive/folders/1qU4cSr3rsSDVRfyVA7IHfGxk-sFUtKIo?usp=sharing) | [Download](https://drive.google.com/drive/folders/1mBn_zq7e0sSH058rEigec3RV1E4nN68X?usp=sharing) |
 
    Herein, we take the initial ranking lists on TREC 2019 DL Track as an example.
    In each link folder, it contains **11** files, one is the original clean one, and 
    others are noisy ones. 
    - Clean: `dl19_43_bm25_1k_Clean_text.tar.gz`, BM25 top-1000 candidates with original clean text.
    - Noisy: `dl19_43_bm25_1k_{noise_type}_text.tar.gz`, 10 separate files with different types of noise. 
    - Format: `q_id \t p_id \t q_text \t p_text \n`
    
### Model:
We release our main re-rankers, they are based on three different backbones and 
trained on original and noisy MS MARCO datasets:
1. BERT-Base:
    
    | BERT_O | De-Ranker_DD | De-Ranker_SD |
    |--------|--------------|--------------|
    | [Download]() | [Download]() | [Download]() |

2. ELECTRA-Base<sup>*</sup>:
    
    | ELECTRA_O | De-Ranker_DD | De-Ranker_SD |
    |--------|--------------|--------------|
    | [Download]() | [Download]() | [Download]() |
    
3. ALBERT-Base<sup>*</sup>:
    
    | ALBERT_O | De-Ranker_DD | De-Ranker_SD |
    |--------|--------------|--------------|
    | [Download]() | [Download]() | [Download]() |

    *The training of our ELECTRA-based and ALBERT-based rerankers is on top of 
    their PyTorch implementations, namely, [electra_pytorch](https://github.com/lonePatient/electra_pytorch)
    and [albert_pytorch](https://github.com/lonePatient/albert_pytorch).
    You need to modify `run_classifier.py` appropriately on the basis of our fine-tuning scripts in this repo.

### Train Triples
The train triples used in our model training: [Train Triples]().
- Format: `query id \t positive id \t negative id \n`
- It contains 400782 train queries and 4170450 train triples.
- Each positive passage is coupled with at most 10 negative passages.


## Citation
If you find our paper/resources useful, please cite:
```
@article{ipm_ChenHHSS23,
  author  = {Xuang Chen and
             Ben He and
             Kai Hui and
             Le Sun and
             Yingfei Sun},
  title   = {Dealing with textual noise for robust and effective BERT re-ranking},
  journal = {Information Processing & Management},
  volume  = {60},
  number  = {1},
  pages   = {103135},
  year    = {2023}
}
```