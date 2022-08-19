import argparse
import random
import spacy
import time


def read_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as qrel:
        for i, line in enumerate(qrel):
            q_id, _, p_id, relevance = line.strip().split()
            assert int(relevance) >= 1
            if q_id not in qrels:
                qrels[q_id] = []
            qrels[q_id].append(p_id)
    return qrels


def read_eval_top(top_file):
    top_dict = {}
    with open(top_file, 'r') as top:
        for line in top:
            q_id, _, p_id, _, _, _ = line.strip().split()
            if q_id not in top_dict:
                top_dict[q_id] = []
            top_dict[q_id].append(p_id)
    return top_dict


def read_corpus(corpus_file):
    p_text_dict = {}
    p_ids_list = []
    with open(corpus_file, 'r') as doc_file:
        for line in doc_file:
            p_id, p_text = line.strip().split('\t')
            if p_id not in p_text_dict:
                p_text_dict[p_id] = p_text.strip()
            else:
                raise KeyError
            p_ids_list.append(p_id)
    p_ids_list = list(set(p_ids_list))
    return p_text_dict, p_ids_list


def read_query(query_file):
    query_dict = {}
    with open(query_file, 'r') as doc_file:
        for line in doc_file:
            qid, qtext = line.strip().split('\t')
            if qid not in query_dict:
                query_dict[qid] = qtext.strip()
            else:
                raise KeyError
    return query_dict


def cal_word_overlap(p1_id, p2_id, p_text_dict, nlp):
    p1_text = p_text_dict[p1_id]
    p2_text = p_text_dict[p2_id]

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()

    p1_word_tokens = word_tokenize(p1_text)
    filtered_p1_word = []
    for w in p1_word_tokens:
        if w not in stop_words:
            filtered_p1_word.append(w)
    p1_words = ' '.join(filtered_p1_word)

    p2_word_tokens = word_tokenize(p2_text)
    filtered_p2_word = []
    for w in p2_word_tokens:
        if w not in stop_words:
            filtered_p2_word.append(w)
    p2_words = ' '.join(filtered_p2_word)

    nlp_p1_words = list(nlp.pipe([p1_words]))[0]
    stem_p1_words = list(set([stemmer.stem(token.norm_.lower()) for token in nlp_p1_words]))

    nlp_p2_words = list(nlp.pipe([p2_words]))[0]
    stem_p2_words = list(set([stemmer.stem(token.norm_.lower()) for token in nlp_p2_words]))

    overlap_num = 0
    for token in stem_p1_words:
        if token in stem_p2_words:
            overlap_num += 1

    overlap_rate = (2 * overlap_num) / (len(stem_p1_words) + len(stem_p2_words))

    return overlap_rate


def add_irrel_text(q_id, p_id, p_text_dict, qrels, top_dict, p_ids_list, nlp, add_place='head'):
    original_p_text = p_text_dict[p_id]

    sampling = True
    irrel_sentences = []
    while sampling:
        random_irrel_p_id = random.sample(p_ids_list, 1)[0]
        overlap_rate = cal_word_overlap(p_id, random_irrel_p_id, p_text_dict, nlp)

        if random_irrel_p_id == p_id:
            continue
        if random_irrel_p_id in qrels[q_id]:
            continue
        if random_irrel_p_id in top_dict[q_id]:
            continue
        if overlap_rate >= 0.05:
            continue

        assert random_irrel_p_id != p_id
        assert random_irrel_p_id not in qrels[q_id]
        assert random_irrel_p_id not in top_dict[q_id]
        assert overlap_rate < 0.05

        irrel_p_text = p_text_dict[random_irrel_p_id]
        irrel_p_text_nlp = list(nlp.pipe([irrel_p_text]))[0]
        irrel_sentences = [str(sent).strip() for sent in irrel_p_text_nlp.sents]

        if len(irrel_sentences) < 3:
            continue

        sampling = False
    assert len(irrel_sentences) >= 3

    add_num = random.sample([1, 2, 3], 1)[0]
    sent_num = len(irrel_sentences)
    sent_index = [[i] for i in range(sent_num)]

    if add_num == 1:
        sample_sent_index = random.sample(sent_index, 1)[0]
    elif add_num == 2:
        sent_index_2 = [[i, i+1] for i in range(sent_num-1)]
        sample_sent_index = random.sample(sent_index_2, 1)[0]
    elif add_num == 3:
        sent_index_3 = [[i, i+1, i+2] for i in range(sent_num-2)]
        sample_sent_index = random.sample(sent_index_3, 1)[0]
    else:
        raise KeyError

    if add_place == 'head':
        modified_p_text = ' '.join([irrel_sentences[i] for i in sample_sent_index]) + ' ' + original_p_text
    elif add_place == 'tail':
        modified_p_text = original_p_text + ' ' + ' '.join([irrel_sentences[i] for i in sample_sent_index])
    elif add_place == 'middle':
        nlp_p = list(nlp.pipe([original_p_text]))[0]
        original_p_sentences = [str(sent).strip() for sent in nlp_p.sents]
        if len(original_p_sentences) == 1:
            modified_p_text = ' '.join([irrel_sentences[i] for i in sample_sent_index]) + ' ' + original_p_text
        else:
            add_point = random.randint(0, len(original_p_sentences) - 2)
            modified_p_text = ' '.join(original_p_sentences[0:add_point + 1]) + ' ' \
                              + ' '.join([irrel_sentences[i] for i in sample_sent_index]) \
                              + ' ' + ' '.join(original_p_sentences[add_point + 1:])
    else:
        raise KeyError

    return modified_p_text


def add_irrel_text_corpus(p_id, p_text_dict, p_ids_list, nlp, add_place='head'):
    original_p_text = p_text_dict[p_id]
    sampling = True
    irrel_sentences = []
    while sampling:
        random_irrel_p_id = random.sample(p_ids_list, 1)[0]
        overlap_rate = cal_word_overlap(p_id, random_irrel_p_id, p_text_dict, nlp)

        if random_irrel_p_id == p_id:
            continue
        else:
            if overlap_rate >= 0.05:
                continue

        assert random_irrel_p_id != p_id
        assert overlap_rate < 0.05

        irrel_p_text = p_text_dict[random_irrel_p_id]
        irrel_p_text_nlp = list(nlp.pipe([irrel_p_text]))[0]
        irrel_sentences = [str(sent).strip() for sent in irrel_p_text_nlp.sents]

        if len(irrel_sentences) < 3:
            continue

        sampling = False
    assert len(irrel_sentences) >= 3

    add_num = random.sample([1, 2, 3], 1)[0]
    sent_num = len(irrel_sentences)
    sent_index = [[i] for i in range(sent_num)]

    if add_num == 1:
        sample_sent_index = random.sample(sent_index, 1)[0]
    elif add_num == 2:
        sent_index_2 = [[i, i+1] for i in range(sent_num-1)]
        sample_sent_index = random.sample(sent_index_2, 1)[0]
    elif add_num == 3:
        sent_index_3 = [[i, i+1, i+2] for i in range(sent_num-2)]
        sample_sent_index = random.sample(sent_index_3, 1)[0]
    else:
        raise KeyError

    if add_place == 'head':
        modified_p_text = ' '.join([irrel_sentences[i] for i in sample_sent_index]) + ' ' + original_p_text
    elif add_place == 'tail':
        modified_p_text = original_p_text + ' ' + ' '.join([irrel_sentences[i] for i in sample_sent_index])
    elif add_place == 'middle':
        nlp_p = list(nlp.pipe([original_p_text]))[0]
        original_p_sentences = [str(sent).strip() for sent in nlp_p.sents]
        if len(original_p_sentences) == 1:
            modified_p_text = ' '.join([irrel_sentences[i] for i in sample_sent_index]) + ' ' + original_p_text
        else:
            add_point = random.randint(0, len(original_p_sentences) - 2)
            modified_p_text = ' '.join(original_p_sentences[0:add_point + 1]) + ' ' \
                              + ' '.join([irrel_sentences[i] for i in sample_sent_index]) \
                              + ' ' + ' '.join(original_p_sentences[add_point + 1:])
    else:
        raise KeyError

    return modified_p_text


def repeat_sents(p_id, p_text_dict, nlp):
    start_time = time.clock()

    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]
    original_p_sentences = [str(sent).strip() for sent in nlp_p.sents]

    sent = ''
    sampling = True
    while sampling:
        sent = random.sample(original_p_sentences, 1)[0]
        if len(sent) < 50 or len(sent) > 300:
            sample_time = time.clock()
            if (sample_time - start_time) > 5:
                sampling = False
            continue
        sampling = False

    repeat_num = random.sample([2, 3], 1)[0]
    sent_index = original_p_sentences.index(sent)
    repeat_p_text = ' '.join(original_p_sentences[0:sent_index]) + ' ' +  (sent + ' ') * repeat_num + \
                    ' '.join(original_p_sentences[sent_index+1:])

    return repeat_p_text


def reorder_sents(p_id, p_text_dict, nlp):
    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]
    original_p_sentences = [str(sent).strip() for sent in nlp_p.sents]

    if len(original_p_sentences) == 1:
        reorder_p_text = original_p_text
    elif len(original_p_sentences) == 2:
        reorder_p_text = original_p_sentences[1] + ' ' + original_p_sentences[0]
    else:
        cut_point = random.randint(0, len(original_p_sentences)-2)
        reorder_p_text = ' '.join(original_p_sentences[cut_point+1:]) + ' ' + ' '.join(original_p_sentences[0:cut_point+1])

    return reorder_p_text


def del_all_punc(p_id, p_text_dict, nlp):
    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]

    del_punc_p_text = ''
    for word in nlp_p:
        if word.pos_ != 'PUNCT':
            if del_punc_p_text == '':
                del_punc_p_text = str(word)
            else:
                del_punc_p_text = del_punc_p_text + ' ' + str(word)
        else:
            continue

    return del_punc_p_text


def add_punc(p_id, p_text_dict, nlp):
    puncs = ['..', '...', '-', '--', '---', '/', '//', '|', '||', '@', '#', '*/', '/*']
    pair_puncs = [['"', '"'],
                  ['{', '}'],
                  ['[', ']'],
                  ['(', ')'],
                  ['`', '`'],
                  ["'", "'"],
                  ['<', '>'],
                  ['|', '|']]

    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]

    modified_sents = []
    for sent in nlp_p.sents:
        words_list = str(sent).split()
        if random.random() < 0.5:
            modified_sents.append(str(sent))
            continue
        else:
            if len(str(sent)) < 50 or len(words_list) <= 5:
                pair_punc = random.sample(pair_puncs, 1)[0]
                sent = pair_punc[0] + ' ' + str(sent) + ' ' + pair_punc[1]
                modified_sents.append(sent)
            else:
                add_num = random.sample([2, 3], 1)[0]
                add_index = random.sample([i for i in range(len(words_list))], add_num)
                new_word_list = []
                for word in words_list:
                    new_word_list.append(word)
                    if words_list.index(word) in add_index:
                        new_word_list.append(random.sample(puncs, 1)[0])
                modified_sents.append(' '.join(new_word_list))
    modified_p_text = ' '.join(modified_sents)

    return modified_p_text


def misspelling(p_id, p_text_dict, nlp):
    from textflint.input_layer.component.sample.sm_sample import SMSample
    from textflint.generation_layer.transformation.UT.keyboard import Keyboard
    keyboard_trans = Keyboard(trans_min=1, trans_max=3)
    from textflint.generation_layer.transformation.UT.ocr import Ocr
    ocr_trans = Ocr(trans_min=1, trans_max=3)

    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]
    trans = [keyboard_trans, ocr_trans]

    modified_sents = []
    for sent in nlp_p.sents:
        if random.random() < 0.5:
            modified_sents.append(str(sent))
            continue
        else:
            data = {'sentence1': '',
                    'sentence2': str(sent).strip(),
                    'y': '0'}
            sample = SMSample(data)

            keyboard_trans_sample = trans[0].transform(sample, field='sentence2', n=1)
            ocr_trans_sample = trans[1].transform(sample, field='sentence2', n=1)
            if len(keyboard_trans_sample) < 1 and len(ocr_trans_sample) == 1:
                modified_sents.append(ocr_trans_sample[0].dump()['sentence2'])
            elif len(keyboard_trans_sample) == 1 and len(ocr_trans_sample) < 1:
                modified_sents.append(keyboard_trans_sample[0].dump()['sentence2'])
            elif len(keyboard_trans_sample) == 1 and len(ocr_trans_sample) == 1:
                trans_sample = [keyboard_trans_sample, ocr_trans_sample]
                modified_sents.append(random.sample(trans_sample, 1)[0][0].dump()['sentence2'])
            else:
                modified_sents.append(str(sent).strip())
    modified_p_text = ' '.join(modified_sents)

    return modified_p_text


def swapsyn(p_id, p_text_dict, nlp):
    from textflint.input_layer.component.sample.sm_sample import SMSample
    from textflint.generation_layer.transformation.UT.swap_syn_wordnet import SwapSynWordNet
    trans = SwapSynWordNet(trans_min=1, trans_max=3)

    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]

    modified_sents = []
    for sent in nlp_p.sents:
        data = {'sentence1': '',
                'sentence2': str(sent).strip(),
                'y': '0'}
        sample = SMSample(data)
        trans_sample = trans.transform(sample, field='sentence2', n=1)
        if len(trans_sample) < 1:
            modified_sents.append(str(sent).strip())
        else:
            modified_sents.append(trans_sample[0].dump()['sentence2'])

    modified_p_text = ' '.join(modified_sents)

    return modified_p_text


def conjoin(p_id, p_text_dict, nlp):
    original_p_text = p_text_dict[p_id]
    nlp_p = list(nlp.pipe([original_p_text]))[0]
    modified_sents = []
    for sent in nlp_p.sents:
        words_list = str(sent).split()
        if random.random() < 0.5:
            modified_sents.append(str(sent))
            continue
        else:
            if len(words_list) < 3:
                modified_sent = ''.join(words_list)
            else:
                conjoin_num = random.sample([2, 3], 1)[0]
                if conjoin_num == 3:
                    conjoin_index = [[i, i + 1, i + 2] for i in range(len(words_list) - 2)]
                elif conjoin_num == 2:
                    conjoin_index = [[i, i + 1] for i in range(len(words_list) - 1)]
                else:
                    raise KeyError
                sample_index = random.sample(conjoin_index, 1)[0]

                modified_sent = ' '.join(words_list[0:sample_index[0]]) + ' ' + \
                                ''.join(words_list[sample_index[0]:sample_index[-1]+1]) + ' ' + \
                                ' '.join(words_list[sample_index[-1]+1:])
            modified_sents.append(modified_sent)

    modified_p_text = ' '.join(modified_sents)

    return modified_p_text


noise_types = ['NrSentH', 'NrSentM', 'NrSentT', 'DupSent', 'RevSent',
               'NoSpace', 'RepSyns', 'ExtraPunc', 'NoPunc', 'MisSpell']


def modify_top_passages(output_pairs_file, top_file, top_dict, p_text_dict, p_ids_list, qrels, query_dict, nlp, noise_type='Random'):

    with open(output_pairs_file, 'w') as out, \
            open(top_file, 'r') as top:
        for i, line in enumerate(top):
            q_id, _, p_id, r, _, _ = line.strip().split()

            if noise_type == 'Random':
                action = random.sample(noise_types, 1)[0]
            else:
                assert noise_type in noise_types
                action = noise_type

            if action == 'NrSentH':
                modified_p_text = add_irrel_text(q_id, p_id, p_text_dict, qrels, top_dict, p_ids_list, nlp, add_place='head')
            elif action == 'NrSentM':
                modified_p_text = add_irrel_text(q_id, p_id, p_text_dict, qrels, top_dict, p_ids_list, nlp, add_place='middle')
            elif action == 'NrSentT':
                modified_p_text = add_irrel_text(q_id, p_id, p_text_dict, qrels, top_dict, p_ids_list, nlp, add_place='tail')
            elif action == 'DupSent':
                modified_p_text = repeat_sents(p_id, p_text_dict, nlp)
            elif action == 'RevSent':
                modified_p_text = reorder_sents(p_id, p_text_dict, nlp)
            elif action == 'NoSpace':
                modified_p_text = conjoin(p_id, p_text_dict, nlp)
            elif action == 'RepSyns':
                modified_p_text = swapsyn(p_id, p_text_dict, nlp)
            elif action == 'ExtraPunc':
                modified_p_text = add_punc(p_id, p_text_dict, nlp)
            elif action == 'NoPunc':
                modified_p_text = del_all_punc(p_id, p_text_dict, nlp)
            elif action == 'MisSpell':
                modified_p_text = misspelling(p_id, p_text_dict, nlp)
            else:
                raise KeyError

            assert modified_p_text is not None
            out.write(q_id + '\t' + p_id + '\t' + query_dict[q_id] + '\t' + modified_p_text + '\n')


def modify_corpus(output_corpus_file, p_text_dict, p_ids_list, nlp):

    with open(output_corpus_file, 'w') as out:
        for p_id in p_ids_list:
            action = random.sample(noise_types, 1)[0]

            if action == 'NrSentH':
                modified_p_text = add_irrel_text_corpus(p_id, p_text_dict, p_ids_list, nlp, add_place='head')
            elif action == 'NrSentM':
                modified_p_text = add_irrel_text_corpus(p_id, p_text_dict, p_ids_list, nlp, add_place='middle')
            elif action == 'NrSentT':
                modified_p_text = add_irrel_text_corpus(p_id, p_text_dict, p_ids_list, nlp, add_place='tail')
            elif action == 'DupSent':
                modified_p_text = repeat_sents(p_id, p_text_dict, nlp)
            elif action == 'RevSent':
                modified_p_text = reorder_sents(p_id, p_text_dict, nlp)
            elif action == 'NoSpace':
                modified_p_text = conjoin(p_id, p_text_dict, nlp)
            elif action == 'RepSyns':
                modified_p_text = swapsyn(p_id, p_text_dict, nlp)
            elif action == 'ExtraPunc':
                modified_p_text = add_punc(p_id, p_text_dict, nlp)
            elif action == 'NoPunc':
                modified_p_text = del_all_punc(p_id, p_text_dict, nlp)
            elif action == 'MisSpell':
                modified_p_text = misspelling(p_id, p_text_dict, nlp)
            else:
                raise KeyError

            out.write(p_id + '\t' + modified_p_text + '\t' + action + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The script to simulate textual noise into top candidates '
                                                 'or into the whole corpus, such as MS MARCO.')
    parser.add_argument('--simulate_mode',
                        default='top',
                        required=True,
                        help='choose in `top or corpus`')
    parser.add_argument('--qrels_file',
                        default=None,
                        help='qrels file for test queries (TREC format).')
    parser.add_argument('--top_file',
                        default=None,
                        help='top recalled initial ranking for test queries, e.g., BM25. (TREC format)')
    parser.add_argument('--corpus_file',
                        default=None,
                        help='corpus file.')
    parser.add_argument('--query_file',
                        default=None,
                        help='test queries')
    parser.add_argument('--output_pairs_file',
                        default=None,
                        help='output noisy top candidates in the form of `q_id \t p_id \t q_text \t noisy_p_text`')
    parser.add_argument('--output_corpus_file',
                        default=None,
                        help="output noisy corpus in the form of `p_id \t noisy_p_text \t noise_type`")
    parser.add_argument('--noise_type',
                        default='Random',
                        help='the single noise type (used for top)')
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')

    if args.simulate_mode == 'top':
        assert args.top_file is not None
        assert args.qrels_file is not None
        assert args.query_file is not None
        assert args.output_pairs_file is not None
        assert args.noise_type is not None
        eval_top_dict = read_eval_top(args.top_file)
        qrels = read_qrels(args.qrels_file)
        query_dict = read_query(args.query_file)
        p_text_dict, p_ids_list = read_corpus(args.corpus_file)
        modify_top_passages(args.output_pairs_file, args.top_file, eval_top_dict, p_text_dict, p_ids_list,
                            qrels, query_dict, nlp, noise_type=args.noise_type)

    elif args.simulate_mode == 'corpus':
        assert args.output_corpus_file is not None
        p_text_dict, p_ids_list = read_corpus(args.corpus_file)
        modify_corpus(args.output_corpus_file, p_text_dict, p_ids_list, nlp)

    else:
        raise NotImplementedError