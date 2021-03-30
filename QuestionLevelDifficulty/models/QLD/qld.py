import math
import torch
import logging
import numpy as np
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from typing import Optional, Dict, List
from models.utils import move_to_device
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BackboneCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lstm = ImageBiLSTM(args)

        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2],
                                      nn.Conv2d(2048, self.args.img_emb_dim, 1))

    def forward(self, src):

        batch_total = torch.tensor([]).to(self.args.device)

        for idx, one_batch in enumerate(src):
            total_data = torch.tensor([])
            for i in range(len(one_batch)):
                temp = one_batch[i].unsqueeze(0)
                total_data = torch.cat([total_data, temp], dim=0)

            total_data = move_to_device(total_data, self.args.device)

            embedding = self.backbone(total_data)
            embedding = embedding.mean(dim=(-2, -1)).unsqueeze(1)
            embedding = self.lstm(embedding)

            batch_total = torch.cat([batch_total, embedding], dim=0)

        return batch_total


class ImageBiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_layers = 1
        self.emb_dim = args.img_emb_dim
        self.hid_dim = args.img_hid_dim

        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers,
                           bidirectional=True, dropout=args.dropout)
        # self.rnn = nn.GRU(self.emb_dim, self.hid_dim, self.n_layers, bidirectional=True, dropout=args.dropout)
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.MaxPool1d(2)
        # self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        # self.avgpool = nn.AvgPool1d(1)
        self.pooling = nn.Sequential(nn.Linear(self.hid_dim * 2, self.hid_dim * 2),
                                     nn.Tanh(),
                                     nn.Dropout(p=args.dropout))

    def forward(self, src):

        # src = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(src)  # lstm
        # outputs, hidden = self.rnn(src)
        bidirectional_hid_1, bidirectional_hid_h2 = hidden[0], hidden[1]
        bidirectional_hid = torch.cat([bidirectional_hid_1, bidirectional_hid_h2], dim=1)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # bidirectional_hid = [n layers, hid dim * n directions], ex) [1, 768]
        # outputs are always from the top hidden layer

        return self.pooling(bidirectional_hid)


class TextEncoder(nn.Module):
    def __init__(self, args, roberta):
        super(TextEncoder, self).__init__()
        self.args = args

        self.roberta = roberta
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self,  enc_inputs: Dict) -> (Tensor, Tensor):

        _, outputs = self.roberta(**enc_inputs)

        return self.dropout(outputs)


class QuestionLevelDifficulty(nn.Module):
    def __init__(self, args, tokenizer, roberta):
        super(QuestionLevelDifficulty, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args
        self.softmax = nn.Softmax(dim=-1)

        self.img_encoder = BackboneCNN(args)
        self.text_encoder = TextEncoder(args, roberta)
        self.attention = DotProductAttention()

        self.memory_class = nn.Linear(768 * 4, 2)  # roberta hidden * memory level | 2, 3
        # self.logic_class = nn.Linear(768 * 4, 4)  # roberta hidden * logic level | 1, 2, 3, 4

    def forward(self,
                text_inputs: Dict,
                img_inputs: Tensor) -> (Tensor, Tensor):

        u = self.img_encoder(img_inputs)
        v = self.text_encoder(text_inputs)

        attn_u = self.attention(v, u, u)
        attn_v = self.attention(u, v, v)

        uv = torch.cat([u, v], dim=1)
        uv = torch.cat([uv, attn_u], dim=1)
        uv = torch.cat([uv, attn_v], dim=1)

        memory_logits = self.softmax(self.memory_class(uv))
        # logic_logits = self.softmax(self.logic_class(uv))

        return memory_logits, 'none' #, logic_logits


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(768)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context