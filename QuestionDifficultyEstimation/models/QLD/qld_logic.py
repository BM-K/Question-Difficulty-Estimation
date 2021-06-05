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

    def forward(self, src):

        batch_total = torch.tensor([]).to(self.args.device)

        for idx, one_batch in enumerate(src):

            embedding = self.lstm(one_batch.unsqueeze(1))

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

        self.pooling = nn.Sequential(nn.Linear(self.hid_dim * 2, self.hid_dim * 2),
                                     nn.Tanh(),
                                     nn.Dropout(p=args.dropout))

    def forward(self, src):

        # src = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(src)
        bidirectional_hid_1, bidirectional_hid_h2 = hidden[0], hidden[1]
        bidirectional_hid = torch.cat([bidirectional_hid_1, bidirectional_hid_h2], dim=1)

        return self.pooling(bidirectional_hid)


class TextEncoder(nn.Module):
    def __init__(self, args, roberta):
        super(TextEncoder, self).__init__()
        self.args = args

        self.roberta = roberta
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self,  enc_inputs: Dict) -> (Tensor, Tensor):

        full, outputs = self.roberta(**enc_inputs)

        return self.dropout(outputs)


class QuestionLevelDifficulty_L(nn.Module):
    def __init__(self, args, tokenizer, roberta):
        super(QuestionLevelDifficulty_L, self).__init__()
    
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args

        self.img_encoder = BackboneCNN(args)
        self.text_encoder = TextEncoder(args, roberta)

        self.logic_class = nn.Linear(768 * 4, 4)
        self.u_attn_layer = nn.ModuleList([TransformerAttn() for _ in range(2)])

    def forward(self,
                text_inputs: Dict,
                img_inputs: Tensor) -> (Tensor, Tensor):

        u = self.img_encoder(img_inputs)
        v = self.text_encoder(text_inputs)

        attn_u = v
        attn_v = u

        for layer in self.u_attn_layer:
            attn_u = layer(attn_u, u, u)

        for layer in self.u_attn_layer:
            attn_v = layer(attn_v, v, v)

        uv = torch.cat([u, v], dim=1)
        uv = torch.cat([uv, attn_u], dim=1)
        uv = torch.cat([uv, attn_v], dim=1)

        outputs = self.logic_class(uv)

        return outputs


class TransformerAttn(nn.Module):
    def __init__(self):
        super(TransformerAttn, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=768, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1)

    def forward(self, query, key, value):

        scores = torch.matmul(query, key.transpose(-1, -2)) #/ np.sqrt(768)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, value)

        output = nn.ReLU()(self.conv1(context.unsqueeze(1).transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)

        return output.squeeze(1)
