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


def write_passage_features(qp_text_file, output_file, tokenizer, max_seq_length):

    with open(qp_text_file, 'r') as text_file, \
            open(output_file, 'w') as csv_file:
        for line in text_file:
            q_id, p_id, q_text, p_text = line.strip().split('\t')
            guid = q_id + '#' + p_id

            input_ids, input_mask, segment_ids = convert_examples_to_features(q_text, p_text, tokenizer, max_seq_length)

            input_ids_str = " ".join([str(id) for id in input_ids])
            input_mask_str = " ".join([str(id) for id in input_mask])
            segment_ids_str = " ".join([str(id) for id in segment_ids])
            csv_file.write(guid + ',' + input_ids_str + ',' + input_mask_str + ',' + segment_ids_str + '\n')


def main():
    parser = ArgumentParser(description='Tokenize the <q_id, p_id, q_text, p_text>')

    parser.add_argument('--max_seq_length', dest='max_seq_length', default=512, type=int)
    parser.add_argument("--qp_text_file",
                        default='./msmarco_passage_data/processed/initial_ranking/msmarco_dev_doct5query_bm25_100_Clean_text.tsv',
                        type=str)
    parser.add_argument("--output_psg_features_file",
                        default='./msmarco_passage_data/processed/initial_ranking/msmarco_dev_doct5query_bm25_100_Clean_text_features.csv',
                        type=str)
    parser.add_argument("--vocab_file",
                        default='./google_bert_models/uncased_L-12_H-768_A-12/vocab.txt',
                        type=str)
    args = parser.parse_args()

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    max_seq_length = args.max_seq_length

    print('Tokenizing Passages...')
    write_passage_features(args.qp_text_file, args.output_psg_features_file, tokenizer, max_seq_length)
    print('Tokenizing Passages done!')


if __name__ == "__main__":
    main()