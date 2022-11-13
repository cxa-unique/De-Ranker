# This code is for data reading

import os
import logging
import torch
import numpy as np
import linecache
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PregeneratedDataset(Dataset):
    def __init__(self, data_path, cache_path, set_type, max_seq_length, num_examples):
        logger.info('data_path: {}'.format(data_path))
        self.seq_len = max_seq_length
        self.set_type = set_type
        self.num_samples = num_examples
        self.working_dir = Path(cache_path)
        input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                              mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
        input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
        segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)

        if self.set_type != 'eval':
            label_ids = np.memmap(filename=self.working_dir/'label_ids.memmap',
                                  shape=(self.num_samples, ), mode='w+', dtype=np.int32)
            label_ids[:] = -1
        else:
            label_ids = None

        logging.info("Loading examples.")

        with open(data_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Examples")):
                tokens = line.strip().split(',')
                guid = tokens[0]
                input_ids[i] = [int(id) for id in tokens[1].split()]
                input_masks[i] = [int(id) for id in tokens[2].split()]
                segment_ids[i] = [int(id) for id in tokens[3].split()]

                if self.set_type != 'eval':
                    label_ids[i] = int(tokens[4])

                    if label_ids[i] != 0 and label_ids[i] != 1:
                        print(i)
                        raise KeyError

                if i < 1:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i]]))
                    logger.info("input_masks: %s" % " ".join([str(x) for x in input_masks[i]]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i]]))
                    if self.set_type != 'eval':
                        logger.info("label: %s" % str(label_ids[i]))

        logging.info("Loading complete!")
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        if self.set_type != 'eval':
            self.label_ids = label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.set_type != 'eval':
            return (torch.tensor(self.input_ids[item], dtype=torch.long),
                    torch.tensor(self.input_masks[item], dtype=torch.long),
                    torch.tensor(self.segment_ids[item], dtype=torch.long),
                    torch.tensor(self.label_ids[item], dtype=torch.long))
        else:
            return (torch.tensor(self.input_ids[item], dtype=torch.long),
                    torch.tensor(self.input_masks[item], dtype=torch.long),
                    torch.tensor(self.segment_ids[item], dtype=torch.long))


def eval_dataloader(args, sampler, batch_size=None):
    file_dir = os.path.join(args.data_dir, args.eval_file_name)

    num_examples = int(len(linecache.getlines(file_dir)))
    print('number of examples: ', str(num_examples))

    dataset = PregeneratedDataset(file_dir, args.cache_file_dir, 'eval', args.max_seq_length, num_examples)

    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return num_examples, dataloader


def train_dataloader(args, sampler, batch_size=None):
    file_dir = os.path.join(args.data_dir, args.train_file_name)

    num_examples = int(len(linecache.getlines(file_dir)))
    print('number of examples: ', str(num_examples))

    dataset = PregeneratedDataset(file_dir, args.cache_file_dir, 'train', args.max_seq_length, num_examples)

    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return num_examples, dataloader


class CrossPregeneratedDataset(Dataset):
    def __init__(self, data_path, cache_path, set_type, max_seq_length, num_examples):
        logger.info('data_path: {}'.format(data_path))
        self.seq_len = max_seq_length
        self.set_type = set_type
        self.num_samples = num_examples
        self.working_dir = Path(cache_path)
        o_input_ids = np.memmap(filename=self.working_dir/'o_input_ids.memmap',
                              mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
        o_input_masks = np.memmap(filename=self.working_dir/'o_input_masks.memmap',
                                shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
        o_segment_ids = np.memmap(filename=self.working_dir/'o_segment_ids.memmap',
                                shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)

        m_input_ids = np.memmap(filename=self.working_dir / 'm_input_ids.memmap',
                                mode='w+', dtype=np.int32, shape=(self.num_samples, self.seq_len))
        m_input_masks = np.memmap(filename=self.working_dir / 'm_input_masks.memmap',
                                  shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)
        m_segment_ids = np.memmap(filename=self.working_dir / 'm_segment_ids.memmap',
                                  shape=(self.num_samples, self.seq_len), mode='w+', dtype=np.int32)

        if self.set_type != 'eval':
            label_ids = np.memmap(filename=self.working_dir/'label_ids.memmap',
                                  shape=(self.num_samples, ), mode='w+', dtype=np.int32)
            label_ids[:] = -1
        else:
            label_ids = None

        logging.info("Loading examples.")

        with open(data_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=self.num_samples, desc="Examples")):
                tokens = line.strip().split(',')
                o_guid = tokens[0]
                o_input_ids[i] = [int(id) for id in tokens[1].split()]
                o_input_masks[i] = [int(id) for id in tokens[2].split()]
                o_segment_ids[i] = [int(id) for id in tokens[3].split()]

                m_guid = tokens[5]
                m_input_ids[i] = [int(id) for id in tokens[6].split()]
                m_input_masks[i] = [int(id) for id in tokens[7].split()]
                m_segment_ids[i] = [int(id) for id in tokens[8].split()]

                assert o_guid in m_guid

                if self.set_type != 'eval':
                    assert int(tokens[4]) == int(tokens[9])
                    label_ids[i] = int(tokens[4])

                    if label_ids[i] != 0 and label_ids[i] != 1:
                        print(i)
                        raise KeyError

                if i < 1:
                    logger.info("*** Example ***")
                    logger.info("guid: %s // %s" % (o_guid, m_guid))
                    logger.info("o_input_ids: %s" % " ".join([str(x) for x in o_input_ids[i]]))
                    logger.info("o_input_masks: %s" % " ".join([str(x) for x in o_input_masks[i]]))
                    logger.info("o_segment_ids: %s" % " ".join([str(x) for x in o_segment_ids[i]]))
                    logger.info("m_input_ids: %s" % " ".join([str(x) for x in m_input_ids[i]]))
                    logger.info("m_input_masks: %s" % " ".join([str(x) for x in m_input_masks[i]]))
                    logger.info("m_segment_ids: %s" % " ".join([str(x) for x in m_segment_ids[i]]))
                    if self.set_type != 'eval':
                        logger.info("label: %s" % str(label_ids[i]))

        logging.info("Loading complete!")
        self.o_input_ids = o_input_ids
        self.o_input_masks = o_input_masks
        self.o_segment_ids = o_segment_ids
        self.m_input_ids = m_input_ids
        self.m_input_masks = m_input_masks
        self.m_segment_ids = m_segment_ids
        if self.set_type != 'eval':
            self.label_ids = label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if self.set_type == 'eval':
            raise NotImplementedError
        else:
            return (torch.tensor(self.o_input_ids[item], dtype=torch.long),
                    torch.tensor(self.o_input_masks[item], dtype=torch.long),
                    torch.tensor(self.o_segment_ids[item], dtype=torch.long),
                    torch.tensor(self.m_input_ids[item], dtype=torch.long),
                    torch.tensor(self.m_input_masks[item], dtype=torch.long),
                    torch.tensor(self.m_segment_ids[item], dtype=torch.long),
                    torch.tensor(self.label_ids[item], dtype=torch.long)
                    )


def train_cross_dataloader(args, sampler, batch_size=None):
    file_dir = os.path.join(args.data_dir, args.train_file_name)

    num_examples = int(len(linecache.getlines(file_dir)))
    print('number of examples: ', str(num_examples))

    dataset = CrossPregeneratedDataset(file_dir, args.cache_file_dir, 'train', args.max_seq_length, num_examples)

    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return num_examples, dataloader