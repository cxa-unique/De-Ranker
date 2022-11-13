#!/bin/bash

data_dir=./beir/datasets
datasets=(scifact climate-fever fever scidocs dbpedia-entity quora webis-touche2020 arguana fiqa hotpotqa nq nfcorpus trec-covid bioasq robust04 cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress)

device=0
model_name=bert_o
model_dir=./msmarco_passage_data/google_bert_base_ft_models/${model_name}/lr3e-6_bs32_epoch1/checkpoint-160000
output_dir=./beir/msmarco_passage_google_bert_base_ft_models/${model_name}/lr3e-6_bs32_epoch1/checkpoint-160000

for ds in ${datasets[@]}
do
  python3 beir_process.py --device ${device} \
                          --dataset ${ds} \
                          --data_dir ${data_dir} \
                          --result_dir ${output_dir} \
                          --model_dir ${model_dir} \
                          --model_name ${model_name}
done