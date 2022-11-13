from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval

from sentence_transformers.cross_encoder import CrossEncoder as CE
from typing import List, Dict, Tuple
import logging
import os
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

### Need to modify index dir and list
index_dir = 'path_to/elasticsearch-7.9.2/data/nodes/0/indices'
index_list = {'scifact': 'wszhmiD7RmOwTjgKNlYuqA',
              'climate-fever': '2-MPqg2JR7aNYz-BizqEag',
              'fever': '2EUyru2ySSWA1BBuDoiqXg',
              'scidocs': 'ICIeJLjLT2Oeq6zixrGbUQ',
              'dbpedia-entity': 'SwAluQWFQSWb92LDsQyJWw',
              'quora': 'CJpc5nZqTNac8UYgDbFdag',
              'webis-touche2020': 'h45Zj-CSRmenw6BCsOVprQ',
              'arguana': 'm-sirQbwQsiWrJzV3GiCpQ',
              'fiqa': 'ohMa4qr5Q4a2jaFf3PCSIQ',
              'hotpotqa': 'sIwqBwO2QYmgNmibpH_DJw',
              'nq': 'C_xeaUlSSuO9jGDLphaTHA',
              'nfcorpus': 'YlYNsRhXTG6H5h7tauwo3A',
              'trec-covid': 'Y8fjo7DpTly7WtAVeaeFhQ',
              'bioasq': 'Ix61eqhTSVKlO9ME-rmyNw',
              'robust04': 'ZjpvK-XaTkaBfPR4DyoRCA',
              'cqadupstack-android': 'hkyFg_QJR_evKvFqd9iWfA', 'cqadupstack-english': 'g0gCD6WCSWe3692GX-MlXA',
              'cqadupstack-gaming': 'Pjikag_NTc20IKnztwbzGw', 'cqadupstack-gis': 'YGnukAgmRSCgdQ5MopGyNw',
              'cqadupstack-mathematica': 'q2C8JLwKRpujzaaY7MP0_w', 'cqadupstack-physics': 'AIDKzTmRRZ6rqSbhvMHwcw',
              'cqadupstack-programmers': '2zWhY3Y9Q2u1LqjX5sbLhQ', 'cqadupstack-stats': '3clPJlt6RKS-b_ePp4MxWQ',
              'cqadupstack-tex': '9LemjoRoQjiSKRxQzoRpFA', 'cqadupstack-unix': 'WgEL3mgBRACOBv-yQXm0sQ',
              'cqadupstack-webmasters': 'h405-KYvTB6rVrUWc69gzQ', 'cqadupstack-wordpress': 'yRSts2aJSf6rJbgWFIkMYg'}


class HomeCrossEncoder:
    def __init__(self, model_path: str = None, max_length: int = None, **kwargs):
        if max_length != None:
            self.model = CE(model_path, max_length=max_length, **kwargs)
        self.model = CE(model_path, **kwargs)

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32, show_progress_bar: bool = True,
                apply_softmax: bool = False, ) -> List[float]:
        return self.model.predict(
            sentences=sentences,
            batch_size=batch_size,
            apply_softmax=apply_softmax,
            show_progress_bar=show_progress_bar)


class HomeRerank:
    def __init__(self, model, batch_size: int = 128, apply_softmax: bool = False, **kwargs):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.apply_softmax = apply_softmax
        self.rerank_results = {}

    def rerank(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:

        sentence_pairs, pair_ids = [], []

        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        rerank_scores = []
        for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size,
                                                apply_softmax=self.apply_softmax):
            rerank_scores.append(float(score[1]))

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default=None,
                        type=str,
                        required=True,
                        help="CUDA device number")
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--result_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True)

    args = parser.parse_args()
    logging.info('The args: {}'.format(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    data_path = "{}/{}".format(args.data_dir, args.dataset)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    #### Provide parameters for elastic-search
    hostname = "localhost"
    index_name = args.dataset.replace('/', '-')
    if index_name in index_list and os.path.exists(os.path.join(index_dir, index_list[index_name])):
        initialize = False
    else:
        initialize = True # True, will delete existing index with same name and reindex all documents
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    ### BM25 run
    results_path = '{}/{}'.format(args.result_dir, args.dataset)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_result_file = results_path + '/run_bm25_elasticsearch_{}_top1k.txt'.format(args.dataset.replace('/', '-'))
    if not os.path.exists(save_result_file):
        with open(save_result_file, 'w') as w:
            for q_id, scores in results.items():
                for i, (doc_id, score) in enumerate(scores.items()):
                    out_str = "{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(q_id, doc_id, i + 1, score, "bm25-elasticsearch")
                    w.write(out_str)

    logging.info('model dir: {}'.format(args.model_dir))
    cross_encoder_model = HomeCrossEncoder(model_path=args.model_dir, num_labels=2)
    reranker = HomeRerank(cross_encoder_model, batch_size=128, apply_softmax=True)

    # Rerank top-100 results using the reranker provided
    rerank_results = reranker.rerank(corpus, queries, results, top_k=100)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

    results_path = '{}/{}/{}'.format(args.result_dir, args.dataset, args.model_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_rerank_result_file = results_path + '/run_bm25_elasticsearch_{}_top1k_rerank.txt'.format(args.dataset.replace('/', '-'))
    with open(save_rerank_result_file, 'w') as w:
        for q_id, scores in rerank_results.items():
            sorted_scores = sorted(scores.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
            for i, (doc_id, score) in enumerate(sorted_scores):
                out_str = "{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(q_id, doc_id, i + 1, score, "bert_rerank")
                w.write(out_str)


if __name__ == "__main__":
    main()