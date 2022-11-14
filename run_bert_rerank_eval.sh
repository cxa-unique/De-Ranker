#!/bin/bash

device=1
cache_file_dir=./cache_bert_rerank_${device}
qrel=path_to/qrels.dev.small.tsv
#qrel=path_to/2019qrels-pass.txt
#qrel=path_to/2020-qrels-pass-final.txt

data_dir=./msmarco_passage_data/processed/initial_ranking
model_dir=./msmarco_passage_data/google_bert_base_ft_models/bert_o/lr3e-6_bs32_epoch1/checkpoint-160000
output_dir=./msmarco_passage_data/google_bert_base_ft_results/bert_o/lr3e-6_bs32_epoch1/checkpoint-160000

noise_types=(Clean NrSentH NrSentM NrSentT DupSent RevSent NoSpace RepSyns ExtraPunc NoPunc MisSpell)
for n_type in ${modified_type[@]}
do
  eval_file_name=msmarco_dev_doct5query_bm25_100_${n_type}_text_features.csv
#  eval_file_name=dl19_43_bm25_1k_${n_type}_text_features.csv
#  eval_file_name=dl20_54_bm25_1k_${n_type}.text_features.csv

  python3 run_bert_rerank_eval.py --device ${device} \
                                  --eval_file_name ${eval_file_name} \
                                  --cache_file_dir ${cache_file_dir} \
                                  --data_dir ${data_dir} \
                                  --output_dir ${output_dir} \
                                  --model_dir ${model_dir} \
                                  --qrels_file ${qrel} \
                                  --eval_batch_size 100 \
                                  --max_seq_length 512
done