import random


def read_corpus(corpus_file):
    print('Loading corpus ...')
    p_text_dict = {}  # {doc_id: doc_text}
    with open(corpus_file, 'r') as doc_file:
        for line in doc_file:
            p_id, p_text = line.strip().split('\t')
            if p_id not in p_text_dict:
                p_text_dict[p_id] = p_text
            else:
                raise KeyError
    print('Loading corpus done!')
    return p_text_dict


def read_query(query_file):
    print('Loading query ...')
    query_dict = {}  # {query_id: query_text}
    with open(query_file, 'r') as doc_file:
        for line in doc_file:
            qid, qtext = line.strip().split('\t')
            if qid not in query_dict:
                query_dict[qid] = qtext.strip()
            else:
                raise KeyError
    print('Loading query done!')
    return query_dict


def read_qrels(qrels_file):
    print('Loading qrels ...')
    qrels = {}
    with open(qrels_file, 'r') as qrel:
        for line in qrel:
            q_id, _, pid, r = line.strip().split('\t')
            assert r == '1'
            if q_id not in qrels:
                qrels[q_id] = []
            qrels[q_id].append(pid)
    print('Loading query done!')
    return qrels


def read_triple_ids(qptriples_file):
    print('Loading triples ...')
    triples = {}
    with open(qptriples_file, 'r') as f:
        for line in f:
            q_id, pos_pid, neg_pid = line.strip().split('\t')
            if q_id not in triples:
                triples[q_id] = {}
            if pos_pid not in triples[q_id]:
                triples[q_id][pos_pid] = []
            triples[q_id][pos_pid].append(neg_pid)
    print('Loading triples done!')
    return triples


## read related data files
p_text_dict = read_corpus('path_to/collection.tsv')
q_text_dict = read_query('path_to/queries.train.tsv')
qp_qrels = read_qrels('path_to/qrels.train.tsv')
qp_triples =  read_triple_ids('path_to/qidpidtriples.train.full.2.tsv')


## sample triples
neg_num = 10
selected_triples = {}
for qid in qp_triples.keys():
    if qid not in selected_triples:
        selected_triples[qid] = []
    for pos in qp_triples[qid].keys():
        assert pos in qp_qrels[qid]
        neg_pids = qp_triples[qid][pos]
        if len(neg_pids) >= neg_num:
            selected_neg_pids = random.sample(neg_pids, neg_num)
        else:
            selected_neg_pids = neg_pids
            print(f'Note that query {qid} has less than {neg_num} negative passages.')

        for selected_neg_pid in selected_neg_pids:
            selected_triples[qid].append((pos, selected_neg_pid))

## save file
with open('./msmarco_passage_data/processed/sampled_train_triples/qidpidtriples.train.full.2.sampled.p1n10.text.tsv', 'w') as out:
    for qid in selected_triples.keys():
        for (p, n) in selected_triples[qid]:
            out.write(qid + '\t' + q_text_dict[qid] + '\t' + p + '\t' + p_text_dict[p] + '\t' + n + '\t' + p_text_dict[n] + '\n')