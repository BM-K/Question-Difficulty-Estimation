import torch
import logging
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, args, roberta):
        super(TextEncoder, self).__init__()
        self.args = args

        self.roberta = roberta
        self.memory_class = nn.Linear(768, 2)  # roberta hidden * memory level | 2, 3
        # self.logic_class = nn.Linear(768, 4)  # roberta hidden * logic level | 1, 2, 3, 4

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self,
                enc_inputs: Dict) -> (Tensor, Tensor):

        _, outputs = self.roberta(**enc_inputs)

        memory_logits = self.memory_class(self.dropout(outputs))

        return memory_logits, 'none'


class QuestionLevelDifficultyOT(nn.Module):
    def __init__(self, args, tokenizer, roberta):
        super(QuestionLevelDifficultyOT, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args
        self.dropout = args.dropout
        self.softmax = nn.Softmax(dim=-1)

        self.text_encoder = TextEncoder(args, roberta)

    def forward(self,
                text_inputs: Dict,
                _: None) -> (Tensor, Tensor):

        memory_logits, logic_logits = self.text_encoder(text_inputs)

        return memory_logits, 'none', 1