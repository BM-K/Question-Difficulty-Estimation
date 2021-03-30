import copy
import torch
import logging
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from models.utils import get_bert_parameter, move_to_device
logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
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
               len(self.attention_mask)

    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')
        vid, qid, question, utterance, memory, logic, shot_c = split_data[0], split_data[1], split_data[2], split_data[4], split_data[5], split_data[6], split_data[8]

        question_tokens = [self.init_token_idx] + self.bert_tokenizer.convert_tokens_to_ids(
             self.bert_tokenizer.tokenize(question)) + [self.sep_token_idx]

        que_utterance_tokens = question_tokens + self.bert_tokenizer.convert_tokens_to_ids(
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
        self.shot_contained.append(shot_c)
        self.segment_ids.append(segment_ids)
        self.attention_mask.append(attention_mask.float())

        return torch.tensor(question), torch.tensor(que_utterance), torch.tensor(int(memory)-2), torch.tensor(int(logic)-1)

    def __getitem__(self, index):
        question_utterance = {'input_ids': self.q_utter[index],
                              'attention_mask': self.attention_mask[index],}
                              #'token_type_ids': self.segment_ids[index]}
        question_utterance = move_to_device(question_utterance, self.args.device)

        get_images_info = {'vid': self.vid[index],
                           'shot': self.shot_contained[index].strip()}

        return question_utterance,\
               get_images_info,\
               self.memory_level[index].to(self.args.device),\
               self.logic_level[index].to(self.args.device),

    def __len__(self):
        return len(self.q_utter)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args):
    path_to_train_data = args.path_to_data+'/'+args.train_data
    path_to_valid_data = args.path_to_data+'/'+args.valid_data
    path_to_test_data = args.path_to_data+'/'+args.test_data

    bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_iter = ModelDataLoader(path_to_train_data, args, bert_tokenizer)
    valid_iter = ModelDataLoader(path_to_valid_data, args, bert_tokenizer)
    test_iter = ModelDataLoader(path_to_test_data, args, bert_tokenizer)

    train_iter.load_data('train')
    valid_iter.load_data('valid')  # 4385
    test_iter.load_data('test')

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'valid': DataLoader(dataset=valid_iter,
                                  batch_size=args.batch_size,
                                  shuffle=True),
              'test': DataLoader(dataset=test_iter,
                                 batch_size=args.batch_size,
                                 shuffle=True)}

    return loader, bert_tokenizer


if __name__ == '__main__':
    get_loader('test')