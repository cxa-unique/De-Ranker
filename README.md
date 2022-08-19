# De-Ranker
This repository contains the code and resources for our paper:
- Dealing with Textual Noise for Robust and Effective BERT Re-ranking. *Under Review*.

## Code
We provide the key script to simulate textual noise into top candidates or 
into the whole corpus: `simulate_textual_noise.py`. 
Before using it, you need to install `TextFlint`, `spaCy` and `NLTK` toolkits. 

Other code and instructions for the training of De-Ranker will release soon.

## Resources
### Synthetic Noisy Data:

1. Noisy version of MS MARCO: [MS MARCO (w/ Noise)](https://drive.google.com/file/d/1nP_ssjGF3g9s_pVLhr5-4FzjiyEw2K9v/view?usp=sharing)
    - It is parallel to original MS MARCO passage corpus: [collection.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz)
    - Format: `p_id \t p_text \t noise_type \n`

2. Noisy Initial Ranking List:

    | MS MARCO Dev | TREC 2019 DL | TREC 2020 DL |
    -----|-----|-----|
    | [Download](https://drive.google.com/drive/folders/1WDJvrwX2AdDj3njtY6dgsbIIsDGftxBr?usp=sharing) | [Download](https://drive.google.com/drive/folders/1qU4cSr3rsSDVRfyVA7IHfGxk-sFUtKIo?usp=sharing) | [Download](https://drive.google.com/drive/folders/1mBn_zq7e0sSH058rEigec3RV1E4nN68X?usp=sharing) |
 
    Herein, we take the initial ranking list on TREC 2019 DL Track as example.
    In each link folder, it contains 11 files, one is the original clean one, and 
    others are noisy ones. 
    - Clean: `dl19_43_bm25_100_Clean_text.tar.gz`
    - Noisy: `dl19_43_bm25_100_{noise_type}_text.tar.gz` (10 separate files with different types of noise). 
    - Format: `q_id \t p_id \t q_text \t p_text \n`
    
### Model:
The trained ranking models will release soon.


### Re-ranking Runs:
The re-ranking run files will release soon.