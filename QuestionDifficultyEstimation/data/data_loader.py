import copy
import h5py
import torch
import logging
import numpy as np
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from models.utils import get_bert_parameter, move_to_device
logger = logging.getLogger(__name__)


class MissO_DataLoader(Dataset):
    def __init__(self, file_path, args, bert_tokenizer):
        self.args = args

        self.qid = []
        self.image = []
        self.q_utter = []
        self.segment_ids = []
        self.logic_level = []
        self.memory_level = []
        self.attention_mask = []
        self.vid = []
        self.shot_contained = []

        self.file_path = file_path
        self.numbering = 1

        self.bert_tokenizer = bert_tokenizer
        self.vocab_size = len(bert_tokenizer)

        """
        init token, idx = <s>, 0
        pad token, idx = <pad>, 1
        sep token, idx = </s>, 2
        unk token, idx = <unk>, 3
        """

        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.init_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token

        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.sep_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.sep_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, data_type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                question, que_utter, memory, logic = self.data2tensor(line)

                self.q_utter.append(que_utter)
                self.memory_level.append(memory)
                self.logic_level.append(logic)

        assert len(self.q_utter) == \
               len(self.memory_level) == \
               len(self.logic_level) == \
               len(self.segment_ids) == \
               len(self.attention_mask) == \
               len(self.image)

    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')
        vid, qid, question, utterance, memory, logic, shot_c, image_path = split_data[0], split_data[1], split_data[2], split_data[4], split_data[5], split_data[6], split_data[8], split_data[9]

        question_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
             self.bert_tokenizer.tokenize(question)) + [self.sep_token_idx]

        if len(utterance) == 0:
            que_utterance_tokens = question_tokens[:-1]
        else:
            if self.args.text_processor == 'bert':
                que_utterance_tokens = question_tokens + self.bert_tokenizer.convert_tokens_to_ids(
                        self.bert_tokenizer.tokenize(utterance)) + [self.sep_token_idx]
            else:
                que_utterance_tokens = question_tokens + [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
                    self.bert_tokenizer.tokenize(utterance)) + [self.sep_token_idx]

        if len(que_utterance_tokens) > self.args.max_len:
            que_utterance_tokens = que_utterance_tokens[:self.args.max_len - 1]
            que_utterance_tokens += [self.sep_token_idx]

        question = copy.deepcopy(question_tokens)
        que_utterance = copy.deepcopy(que_utterance_tokens)

        for i in range(self.args.max_len - len(question_tokens)):question.append(self.pad_token_idx)
        for i in range(self.args.max_len - len(que_utterance_tokens)):que_utterance.append(self.pad_token_idx)

        segment_ids, attention_mask = get_bert_parameter(que_utterance,
                                                         self.pad_token_idx,
                                                         self.sep_token_idx,
                                                         self.args)

        self.vid.append(vid)
        self.qid.append(qid)
        self.segment_ids.append(segment_ids)
        self.image.append(image_path.strip())
        self.attention_mask.append(attention_mask.float())

        return torch.tensor(question), torch.tensor(que_utterance), torch.tensor(int(memory)-2), torch.tensor(int(logic)-1)

    def __getitem__(self, index):
        question_utterance = {'input_ids': self.q_utter[index],
                              'attention_mask': self.attention_mask[index],}
                              #'token_type_ids': self.segment_ids[index]}
        question_utterance = move_to_device(question_utterance, self.args.device)

        get_images_info = {'vid': self.vid[index],
                           'qid': str(self.qid[index]),
                           'path_to_images': self.image[index]}

        return question_utterance,\
               get_images_info,\
               self.memory_level[index].to(self.args.device),\
               self.logic_level[index].to(self.args.device),

    def __len__(self):
        return len(self.q_utter)


class TVQA_DataLoader(Dataset):
    def __init__(self, file_path, args, bert_tokenizer):
        self.args = args

        self.vid = []
        self.image = []
        self.question = []
        self.que_sub_text = []
        self.diff_level = []

        self.segment_ids = []
        self.attention_mask = []

        self.file_path = file_path
        self.numbering = 1

        self.bert_tokenizer = bert_tokenizer
        self.vocab_size = len(bert_tokenizer)

        """ roberta
        init token, idx = <s>, 0
        pad token, idx = <pad>, 1
        sep token, idx = </s>, 2
        unk token, idx = <unk>, 3
        """

        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.init_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token

        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.sep_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.sep_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, data_type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                question, que_sub_text, level, = self.data2tensor(line)

                self.question.append(question)
                self.diff_level.append(level)
                self.que_sub_text.append(que_sub_text)

        assert len(self.question) == \
               len(self.diff_level) == \
               len(self.que_sub_text) == \
               len(self.segment_ids) == \
               len(self.attention_mask)

    def preprocessing(self, sentence):
        return sentence.lower()
    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')
        question, level, sub_text, vid = split_data[0], split_data[1], split_data[2], split_data[3]

        question_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
             self.bert_tokenizer.tokenize(question)) + [self.sep_token_idx]

        if len(sub_text) == 0:
            que_utterance_tokens = question_tokens[:-1]
        else:
            if self.args.text_processor == 'bert':
                que_utterance_tokens = question_tokens + self.bert_tokenizer.convert_tokens_to_ids(
                        self.bert_tokenizer.tokenize(sub_text)) + [self.sep_token_idx]
            else:
                que_utterance_tokens = question_tokens + [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
                    self.bert_tokenizer.tokenize(sub_text)) + [self.sep_token_idx]

        if len(que_utterance_tokens) > self.args.max_len:
            que_utterance_tokens = que_utterance_tokens[:self.args.max_len - 1]
            que_utterance_tokens += [self.sep_token_idx]

        question = copy.deepcopy(question_tokens)
        que_utterance = copy.deepcopy(que_utterance_tokens)

        for i in range(self.args.max_len - len(question_tokens)):question.append(self.pad_token_idx)
        for i in range(self.args.max_len - len(que_utterance_tokens)):que_utterance.append(self.pad_token_idx)

        segment_ids, attention_mask = get_bert_parameter(que_utterance,
                                                         self.pad_token_idx,
                                                         self.sep_token_idx,
                                                         self.args)

        self.segment_ids.append(segment_ids)
        self.attention_mask.append(attention_mask.float())
        self.image.append(vid)

        return torch.tensor(question), torch.tensor(que_utterance), torch.tensor(int(level))

    def __getitem__(self, index):
        question_utterance = {'input_ids': self.que_sub_text[index],
                              'attention_mask': self.attention_mask[index],}
                              #'token_type_ids': self.segment_ids[index]}
        question_utterance = move_to_device(question_utterance, self.args.device)

        if self.args.only_text_input == 'False':
            return question_utterance,\
                   self.image[index],\
                   self.diff_level[index].to(self.args.device), \
                   torch.tensor([0])
        else:
            return question_utterance, \
                   torch.tensor([0]), \
                   self.diff_level[index].to(self.args.device), \
                   torch.tensor([0])

    def __len__(self):
        return len(self.question)


class TVQA_DataLoader_split(Dataset):
    def __init__(self, file_path, args, bert_tokenizer):
        self.args = args

        self.vid = []
        self.image = []
        self.question = []
        self.uttr = []
        self.diff_level = []

        self.segment_ids = []
        self.attention_mask_utt = []
        self.attention_mask_que = []

        self.file_path = file_path
        self.numbering = 1

        self.bert_tokenizer = bert_tokenizer
        self.vocab_size = len(bert_tokenizer)

        """ roberta
        init token, idx = <s>, 0
        pad token, idx = <pad>, 1
        sep token, idx = </s>, 2
        unk token, idx = <unk>, 3
        """

        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.init_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token

        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.sep_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.sep_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, data_type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                question, uttr, level, = self.data2tensor(line)

                self.question.append(question)
                self.diff_level.append(level)
                self.uttr.append(uttr)

        assert len(self.question) == \
               len(self.diff_level) == \
               len(self.uttr) == \
               len(self.segment_ids)

    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')
        question, level, sub_text, vid = split_data[0], split_data[1], split_data[2], split_data[3]

        question_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
             self.bert_tokenizer.tokenize(question)) + [self.sep_token_idx]


        if self.args.text_processor == 'bert':
            que_utterance_tokens = question_tokens + self.bert_tokenizer.convert_tokens_to_ids(
                    self.bert_tokenizer.tokenize(sub_text)) + [self.sep_token_idx]
        else:
            if len(sub_text) == 0:
                utterance_tokens = [self.init_token_idx] + [self.sep_token_idx]
            else:
                utterance_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
                    self.bert_tokenizer.tokenize(sub_text)) + [self.sep_token_idx]

        if len(utterance_tokens) > self.args.max_len:
            utterance_tokens = utterance_tokens[:self.args.max_len - 1]
            utterance_tokens += [self.sep_token_idx]

        question = copy.deepcopy(question_tokens)
        que_utterance = copy.deepcopy(utterance_tokens)

        for i in range(self.args.max_len - len(question_tokens)):question.append(self.pad_token_idx)
        for i in range(self.args.max_len - len(utterance_tokens)):que_utterance.append(self.pad_token_idx)

        assert len(question) == len(que_utterance)

        segment_ids, attention_mask_utt = get_bert_parameter(que_utterance,
                                                             self.pad_token_idx,
                                                             self.sep_token_idx,
                                                             self.args)
        segment_ids, attention_mask_que = get_bert_parameter(question,
                                                             self.pad_token_idx,
                                                             self.sep_token_idx,
                                                             self.args)

        self.segment_ids.append(segment_ids)
        self.attention_mask_utt.append(attention_mask_utt.float())
        self.attention_mask_que.append(attention_mask_que.float())
        self.image.append(vid)

        return torch.tensor(question), torch.tensor(que_utterance), torch.tensor(int(level))

    def __getitem__(self, index):
        text_inputs = []

        utterance = {'input_ids': self.uttr[index],
                     'attention_mask': self.attention_mask_utt[index],}
                              #'token_type_ids': self.segment_ids[index]}
        utterance = move_to_device(utterance, self.args.device)

        que = {'input_ids': self.question[index],
                     'attention_mask': self.attention_mask_que[index], }
        # 'token_type_ids': self.segment_ids[index]}
        que = move_to_device(que, self.args.device)

        text_inputs.append(que)
        text_inputs.append(utterance)

        if self.args.only_text_input == 'False':
            return text_inputs,\
                   self.image[index],\
                   self.diff_level[index].to(self.args.device), \
                   torch.tensor([0])
        else:
            return text_inputs, \
                   torch.tensor([0]), \
                   self.diff_level[index].to(self.args.device), \
                   torch.tensor([0])

    def __len__(self):
        return len(self.question)


class MissO_DataLoader_split_text(Dataset):
    def __init__(self, file_path, args, bert_tokenizer):
        self.args = args

        self.qid = []
        self.image = []

        self.que = []
        self.utter = []

        self.segment_ids = []

        self.logic_level = []
        self.memory_level = []

        self.attention_mask_que = []
        self.attention_mask_utt = []

        self.vid = []
        self.shot_contained = []

        self.file_path = file_path
        self.numbering = 1

        self.bert_tokenizer = bert_tokenizer
        self.vocab_size = len(bert_tokenizer)

        """
        init token, idx = <s>, 0
        pad token, idx = <pad>, 1
        sep token, idx = </s>, 2
        unk token, idx = <unk>, 3
        """

        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.init_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token

        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.sep_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.sep_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, data_type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                question, utter, memory, logic = self.data2tensor_text_split(line)

                self.que.append(question)
                self.utter.append(utter)
                self.memory_level.append(memory)
                self.logic_level.append(logic)

        assert len(self.utter) == \
               len(self.memory_level) == \
               len(self.logic_level) == \
               len(self.segment_ids) == \
               len(self.attention_mask_que) == \
               len(self.image)

    def data2tensor_text_split(self, line):
        split_data = line.split('\t')
        vid, qid, question, utterance, memory, logic, shot_c, image_path = split_data[0], split_data[1], split_data[
            2], split_data[4], split_data[5], split_data[6], split_data[8], split_data[9]

        question_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
            self.bert_tokenizer.tokenize(question)) + [self.sep_token_idx]

        if self.args.text_processor == 'bert':
            que_utterance_tokens = question_tokens + self.bert_tokenizer.convert_tokens_to_ids(
                self.bert_tokenizer.tokenize(utterance)) + [self.sep_token_idx]
        else:
            if len(utterance) == 0:
                utterance_tokens = [self.init_token_idx] + [self.sep_token_idx]
            else:
                utterance_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
                    self.bert_tokenizer.tokenize(utterance)) + [self.sep_token_idx]

        if len(utterance_tokens) > self.args.max_len:
            utterance_tokens = utterance_tokens[:self.args.max_len - 1]
            utterance_tokens += [self.sep_token_idx]

        question = copy.deepcopy(question_tokens)
        que_utterance = copy.deepcopy(utterance_tokens)

        for i in range(self.args.max_len - len(question_tokens)): question.append(self.pad_token_idx)
        for i in range(self.args.max_len - len(utterance_tokens)): que_utterance.append(self.pad_token_idx)

        assert len(question) == len(que_utterance)

        segment_ids, attention_mask_utt = get_bert_parameter(que_utterance,
                                                             self.pad_token_idx,
                                                             self.sep_token_idx,
                                                             self.args)
        segment_ids, attention_mask_que = get_bert_parameter(question,
                                                             self.pad_token_idx,
                                                             self.sep_token_idx,
                                                             self.args)

        self.vid.append(vid)
        self.qid.append(qid)
        self.segment_ids.append(segment_ids)
        self.image.append(image_path.strip())

        self.attention_mask_que.append(attention_mask_que.float())
        self.attention_mask_utt.append(attention_mask_utt.float())

        return torch.tensor(question), torch.tensor(que_utterance), torch.tensor(int(memory) - 2), torch.tensor(
            int(logic) - 1)

    def __getitem__(self, index):
        text_inputs = []

        utterance = {'input_ids': self.utter[index],
                     'attention_mask': self.attention_mask_utt[index],}

        utterance = move_to_device(utterance, self.args.device)

        que = {'input_ids': self.que[index],
               'attention_mask': self.attention_mask_que[index], }

        que = move_to_device(que, self.args.device)

        get_images_info = {'vid': self.vid[index],
                           'qid': str(self.qid[index]),
                           'path_to_images': self.image[index]}

        text_inputs.append(que)
        text_inputs.append(utterance)

        return text_inputs,\
               get_images_info,\
               self.memory_level[index].to(self.args.device),\
               self.logic_level[index].to(self.args.device), \

    def __len__(self):
        return len(self.utter)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args):
    path_to_train_data = args.path_to_data+'/'+args.train_data
    path_to_valid_data = args.path_to_data+'/'+args.valid_data
    path_to_test_data = args.path_to_data+'/'+args.test_data

    if args.text_processor == 'roberta':
        bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.text_processor == 'bert':
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        bert_tokenizer = None

    if args.dataset == 'MissO':
        train_iter = MissO_DataLoader(path_to_train_data, args, bert_tokenizer)
        valid_iter = MissO_DataLoader(path_to_valid_data, args, bert_tokenizer)
        test_iter = MissO_DataLoader(path_to_test_data, args, bert_tokenizer)
    elif args.dataset == 'TVQA':
        train_iter = TVQA_DataLoader(path_to_train_data, args, bert_tokenizer)
        valid_iter = TVQA_DataLoader(path_to_valid_data, args, bert_tokenizer)
        test_iter = TVQA_DataLoader(path_to_test_data, args, bert_tokenizer)
    elif args.dataset == 'TVQA_split':
        train_iter = TVQA_DataLoader_split(path_to_train_data, args, bert_tokenizer)
        valid_iter = TVQA_DataLoader_split(path_to_valid_data, args, bert_tokenizer)
        test_iter = TVQA_DataLoader_split(path_to_test_data, args, bert_tokenizer)
    elif args.dataset == 'MissO_split':
        train_iter = MissO_DataLoader_split_text(path_to_train_data, args, bert_tokenizer)
        valid_iter = MissO_DataLoader_split_text(path_to_valid_data, args, bert_tokenizer)
        test_iter = MissO_DataLoader_split_text(path_to_test_data, args, bert_tokenizer)
    else:
        print("Data set name ERROR")
        exit()

    train_iter.load_data('train')
    valid_iter.load_data('valid')
    test_iter.load_data('test')

    batch_size = args.batch_size

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=batch_size,
                                  shuffle=True),
              'valid': DataLoader(dataset=valid_iter,
                                  batch_size=batch_size,
                                  shuffle=True),
              'test': DataLoader(dataset=test_iter,
                                 batch_size=batch_size,
                                 shuffle=True)}

    return loader, bert_tokenizer


if __name__ == '__main__':
    get_loader('test')
