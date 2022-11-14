#!/bin/bash

bs=32
ep=1
lr=3e-6

data_dir=./msmarco_passage_data/processed/sampled_train_triples
model_dir=./msmarco_passage_data/google_bert_base_ft_models/bert_o/lr3e-6_bs32_epoch1/checkpoint-160000

train_features_file=qidpidtriples.train.full.2.sampled.p1n10.origin-noise.features.csv
cache_dir=./cache/cache_deranker-sd_lr${lr}_bs${bs}_epoch${ep}
output_dir=./msmarco_passage_data/google_bert_base_ft_models/deranker_sd/lr${lr}_bs${bs}_epoch${ep}

CUDA_VISIBLE_DEVICES=0,1,2 python3 deranker_finetune_sd.py --data_dir ${data_dir} \
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