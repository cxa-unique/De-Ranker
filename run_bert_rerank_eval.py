import os
import logging
import argparse
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import SequentialSampler
from bert.modeling import BertForSequenceClassification
from features_csv_reader import eval_dataloader
from dl_trec_eval import evaluate_metrics
from msmarco_mrr_eval import cal_mrr


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def do_eval(model, eval_dataloader, device):
    scores = []
    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids = batch_
            logits = model(input_ids, segment_ids, input_mask)

            probs = F.softmax(logits, dim=1)[:, 1]
            scores.append(probs.detach().cpu().numpy())

    result = {}
    result['scores'] = np.concatenate(scores)

    return result


def get_trec_result(args, scores):

    query_docids_map = []
    with open(os.path.join(args.data_dir, args.eval_file_name)) as ref_file:
        for line in ref_file:
            q_id, p_id = line.strip().split(",")[0].split('#')
            if '-' in p_id:
                p_id = p_id.split('-')[0]
            query_docids_map.append([q_id, p_id])

    trec_file_dir = os.path.join(args.output_dir, "{0}_rerank.trec".format(args.eval_file_name))

    assert len(scores) == len(query_docids_map)

    ranking = {}
    for idx, score in enumerate(scores):
        query_id, doc_id = query_docids_map[idx]
        if query_id not in ranking:
            ranking[query_id] = []
        ranking[query_id].append((float(score), doc_id))

    with open(trec_file_dir, 'w') as trec_file:
        for qid in ranking.keys():
            sorted_ranking = sorted(ranking[qid], reverse=True)
            for rank, item in enumerate(sorted_ranking):
                score, docid = item
                out_str = "{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(qid, docid, rank + 1, score, "bert_rerank")
                trec_file.write(out_str)

    logger.info("Done Evaluating!")

    return trec_file_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        required=True,
                        help="CUDA device number")
    parser.add_argument("--eval_file_name",
                        default=None,
                        type=str,
                        help="The features file for query-passage pairs.")
    parser.add_argument("--cache_file_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The cache dir used for data reading.")
    parser.add_argument("--qrels_file",
                        default=None,
                        type=str,
                        help="The annotation file.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the re-ranking runs will be written.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        help="The model dir, which contains the model waited for evaluation.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size for eval.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenization.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.cache_file_dir):
        os.makedirs(args.cache_file_dir)

    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=2)
    model.to(device)

    _, dataloader = eval_dataloader(args, SequentialSampler, args.eval_batch_size)
    dev_res = do_eval(model, dataloader, device)
    dev_scores = dev_res['scores']
    trec_file_dir = get_trec_result(args, dev_scores)

    if 'dev' in args.qrels_file:
        query_num, mrr, mrr_ten, _ = cal_mrr(args.qrels_file, trec_file_dir)
        print('Eval: query_num: {}, MRR: {}, MRR@10: {}'.format(query_num, mrr, mrr_ten))
    else:
        metrics_dict = evaluate_metrics(args.qrels_file, trec_file_dir,
                                        ['recip_rank', 'ndcg_cut_10', 'map', 'recall_100', 'recall_500', 'recall_1000'])
        print('Eval: {}'.format(metrics_dict))

    if os.path.exists(args.cache_file_dir):
        import shutil
        shutil.rmtree(args.cache_file_dir)


if __name__ == "__main__":
    main()
