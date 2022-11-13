#!/bin/bash

bs=32
ep=1
lr=3e-6
data_dir=./msmarco_passage_data/processed/sampled_train_triples
model_dir=./google_bert_models/uncased_L-12_H-768_A-12

## BERT_O
train_features_file=qidpidtriples.train.full.2.sampled.p1n10.features.csv
output_dir=./msmarco_passage_data/google_bert_base_ft_models/bert_o/lr${lr}_bs${bs}_epoch${ep}
cache_dir=./cache/cache_bert-o_lr${lr}_bs${bs}_epoch${ep}

## BERT_O+N
#noise_rate=0.5
#train_features_file=qidpidtriples.train.full.2.sampled.p1n10.noise-mix-${noise_rate}.shuf.features.csv
#output_dir=./msmarco_passage_data/google_bert_base_ft_models/bert_o_${noise_rate}n/lr${lr}_bs${bs}_epoch${ep}
#cache_dir=./cache/cache_bert_o_${noise_rate}n_lr${lr}_bs${bs}_epoch${ep}

CUDA_VISIBLE_DEVICES=0,1 python3 vanilla_bert_finetune.py --data_dir ${data_dir} \
                                                          --model ${model_dir} \
                                                          --output_dir ${output_dir} \
                                                          --train_file_name ${train_features_file} \
                                                          --cache_file_dir ${cache_dir} \
                                                          --max_seq_length 512 \
                                                          --do_lower_case \
                                                          --train_batch_size ${bs} \
                                                          --learning_rate ${lr} \
                                                          --num_train_epochs ${ep}.0 \
                                                          --seed 42 \
                                                          --eval_step 1000 \
                                                          --save_step 10000 \
                                                          --fp16