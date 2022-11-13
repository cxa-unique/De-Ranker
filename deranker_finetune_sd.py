"""De-Ranker static-denoising finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import RandomSampler
from tqdm import tqdm, trange

from bert.modeling import BertForSequenceClassification
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam
from features_csv_reader import train_cross_dataloader

CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        for key in result.keys():
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("-----------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True,
                        help="The student model dir.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_file_name",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--cache_file_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)
    parser.add_argument('--save_step',
                        type=int,
                        default=5000)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.cache_file_dir):
        os.makedirs(args.cache_file_dir)

    # Prepare Data
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    original_model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2)
    original_model.to(device)
    model.to(device)

    num_examples, dataloader = train_cross_dataloader(args, RandomSampler, batch_size=args.train_batch_size)

    num_train_optimization_steps = int(
        num_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
        if 'classifier' in n:
            p.requires_grad = False
            logger.info('p: {}'.format(p))

    logger.info('Total parameters of student_model: {}'.format(size))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in param_optimizer if 'classifier' not in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                         schedule=schedule,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        logger.info('FP16 is activated, use amp')
    else:
        logger.info('FP16 is not activated, only use BertAdam')

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        original_model = torch.nn.DataParallel(original_model)

    # Train
    global_step = 0
    tr_loss = 0.
    tr_o_mse_loss = 0.
    tr_m_mse_loss = 0.
    output_loss_file = os.path.join(args.output_dir, "train_loss.txt")
    mse_loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)

            o_input_ids, o_input_mask, o_segment_ids, m_input_ids, m_input_mask, m_segment_ids, label_ids = batch
            batch_size = o_input_ids.size(0)

            if o_input_ids.size()[0] != args.train_batch_size:
                continue

            with torch.no_grad():
                _, original_pooled_cls = original_model(o_input_ids, o_segment_ids, o_input_mask, return_pooled_cls=True)

            input_ids = torch.cat((o_input_ids, m_input_ids), dim=0)
            segment_ids = torch.cat((o_segment_ids, m_segment_ids), dim=0)
            input_mask = torch.cat((o_input_mask, m_input_mask), dim=0)
            _, pooled_cls = model(input_ids, segment_ids, input_mask, return_pooled_cls=True)
            o_pooled_cls = pooled_cls[0: batch_size, :]
            m_pooled_cls = pooled_cls[batch_size:, :]

            mse_loss_o = mse_loss_fn(o_pooled_cls, original_pooled_cls)
            mse_loss_m = mse_loss_fn(m_pooled_cls, original_pooled_cls)
            loss = mse_loss_o + mse_loss_m

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_o_mse_loss += mse_loss_o.item()
            tr_m_mse_loss += mse_loss_m.item()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.eval_step == 0:
                print_loss = tr_loss / args.eval_step
                print_o_mse_loss = tr_o_mse_loss / args.eval_step
                print_m_mse_loss = tr_m_mse_loss / args.eval_step
                result = {}
                result['global_step'] = global_step
                result['o_mse_loss'] = print_o_mse_loss
                result['m_mse_loss'] = print_m_mse_loss
                result['loss'] = print_loss
                result_to_file(result, output_loss_file)

                tr_loss = 0.
                tr_o_mse_loss = 0.
                tr_m_mse_loss = 0.

            if global_step % args.save_step == 0:
                logger.info("***** Save model *****")
                model_to_save = model.module if hasattr(model, 'module') else model
                model_name = WEIGHTS_NAME
                checkpoint_name = 'checkpoint-' + str(global_step)
                output_model_dir = os.path.join(args.output_dir, checkpoint_name)
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)
                output_model_file = os.path.join(output_model_dir, model_name)
                output_config_file = os.path.join(output_model_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_model_dir)

    logger.info("***** Save model *****")
    model_to_save = model.module if hasattr(model, 'module') else model
    model_name = WEIGHTS_NAME
    checkpoint_name = 'checkpoint-' + str(global_step)
    output_model_dir = os.path.join(args.output_dir, checkpoint_name)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    output_model_file = os.path.join(output_model_dir, model_name)
    output_config_file = os.path.join(output_model_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_model_dir)

    if os.path.exists(args.cache_file_dir):
        import shutil
        shutil.rmtree(args.cache_file_dir)

if __name__ == "__main__":
    main()
