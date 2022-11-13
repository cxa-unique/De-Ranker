from argparse import ArgumentParser
import tokenization


def convert_examples_to_features(q_text, p_text, tokenizer, max_seq_length):
    query = tokenization.convert_to_unicode(q_text)
    query_tokens = tokenization.convert_to_bert_input(
        text=query,
        max_seq_length=128,
        tokenizer=tokenizer,
        add_cls=True,
        add_sep=True,
        truncate_warning=True)

    psg = tokenization.convert_to_unicode(p_text)
    psg_tokens = tokenization.convert_to_bert_input(
        text=psg,
        max_seq_length=(max_seq_length - len(query_tokens) + 1),
        tokenizer=tokenizer,
        add_cls=False,
        add_sep=True,
        truncate_warning=True)

    assert query_tokens[0] == 101
    assert query_tokens[-1] == 102
    assert psg_tokens[-1] == 102

    input_ids = query_tokens + psg_tokens
    segment_ids = [0] * len(query_tokens) + [1] * len(psg_tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def write_passage_features(train_triples_file, output_file, tokenizer, max_seq_length):
    with open(train_triples_file, 'r') as triples, \
            open(output_file, 'w') as csv_file:
        for i, line in enumerate(triples):
            # q_id, q_text, pos_p_id, pos_p_text, neg_p_id, neg_p_text, modify_type = line.strip().split('\t')
            q_id, q_text, pos_p_id, pos_p_text, neg_p_id, neg_p_text = line.strip().split('\t')
            # pos_guid = '{}-pos-{}-{}'.format(q_id, pos_p_id, modify_type.strip('#'))
            # neg_guid = '{}-neg-{}-{}'.format(q_id, neg_p_id, modify_type.strip('#'))
            pos_guid = '{}-pos-{}'.format(q_id, pos_p_id)
            neg_guid = '{}-neg-{}'.format(q_id, neg_p_id)

            pos_input_ids, pos_input_mask, pos_segment_ids = convert_examples_to_features(q_text, pos_p_text, tokenizer, max_seq_length)
            pos_input_ids_str = " ".join([str(id) for id in pos_input_ids])
            pos_input_mask_str = " ".join([str(id) for id in pos_input_mask])
            pos_segment_ids_str = " ".join([str(id) for id in pos_segment_ids])
            csv_file.write(pos_guid + ',' + pos_input_ids_str + ',' + pos_input_mask_str + ',' + pos_segment_ids_str + ',' + '1' + '\n')

            neg_input_ids, neg_input_mask, neg_segment_ids = convert_examples_to_features(q_text, neg_p_text, tokenizer, max_seq_length)
            neg_input_ids_str = " ".join([str(id) for id in neg_input_ids])
            neg_input_mask_str = " ".join([str(id) for id in neg_input_mask])
            neg_segment_ids_str = " ".join([str(id) for id in neg_segment_ids])
            csv_file.write(neg_guid + ',' + neg_input_ids_str + ',' + neg_input_mask_str + ',' + neg_segment_ids_str + ',' + '0' + '\n')


def main():
    parser = ArgumentParser(description='Tokenize the <qid, query, pos_id, pos_text, neg_id, neg_text>')

    parser.add_argument('--max_seq_length', dest='max_seq_length', default=512, type=int)
    parser.add_argument("--train_triples_file",
                        default='./msmarco_passage_data/processed/sampled_train_triples/qidpidtriples.train.full.2.sampled.p1n10.text.tsv',
                        type=str)
    parser.add_argument("--output_train_features_file",
                        default='./msmarco_passage_data/processed/sampled_train_triples/qidpidtriples.train.full.2.sampled.p1n10.features.csv',
                        type=str)
    parser.add_argument("--vocab_file",
                        default='./google_bert_models/uncased_L-12_H-768_A-12/vocab.txt',
                        type=str)
    args = parser.parse_args()

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    max_seq_length = args.max_seq_length

    print('Tokenizing Triples...')
    write_passage_features(args.train_triples_file, args.output_train_features_file, tokenizer, max_seq_length)
    print('Tokenizing Triples done!')


if __name__ == "__main__":
    main()