import math
import torch
import logging
import numpy as np
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from typing import Optional, Dict, List
from models.utils import move_to_device

logger = logging.getLogger(__name__)


class BackboneCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lstm = ImageBiLSTM(args)
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2],
                                      nn.Conv2d(2048, self.args.img_emb_dim, 1))

        self.MVQA_linear = nn.Linear(2048, self.args.img_emb_dim)

    def forward(self, src):

        batch_total = torch.tensor([]).to(self.args.device)

        if self.args.dataset == 'MissO' or self.args.dataset == 'MissO_split':

            for idx, one_batch in enumerate(src):

                embedding = self.lstm(one_batch.unsqueeze(1))

                batch_total = torch.cat([batch_total, embedding], dim=0)

        elif self.args.dataset == 'TVQA' or self.args.dataset == 'TVQA_split':

            for idx, one_batch in enumerate(src):

                embedding = self.lstm(self.MVQA_linear(one_batch).unsqueeze(1))

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

    def forward(self, enc_inputs: Dict) -> (Tensor, Tensor):

        full, outputs = self.roberta(**enc_inputs)

        return self.dropout(outputs)


class Bilstm_HAM(nn.Module):
    def __init__(self, args):
        super(Bilstm_HAM, self).__init__()
        self.args = args
        self.n_layers = 1
        self.emb_dim = 768 * 3
        self.hid_dim = int(768 / 2)
        self.maxpool = nn.MaxPool1d(2)

        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers,
                           bidirectional=True, dropout=args.dropout)

        self.li = nn.Linear(int(768 / 2), int(768 / 2))
        self.fc = nn.Linear(int(768 / 2), 2)

    def forward(self, src):

        # src = [src len, batch size, emb dim]
        src = src.unsqueeze(0)
        outputs, (hidden, cell) = self.rnn(src)
        outputs = self.maxpool(outputs).squeeze(0)
        outputs = self.fc(nn.ReLU()(self.li(outputs)))

        return outputs


class Bilstm_OUR(nn.Module):
    def __init__(self, args):
        super(Bilstm_OUR, self).__init__()
        self.args = args
        self.n_layers = 1
        self.emb_dim = 768 * 2
        self.hid_dim = 768

        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers,
                           bidirectional=True, dropout=args.dropout)

        self.li = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 2)
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, src):

        # src = [src len, batch size, emb dim]
        src = src.unsqueeze(0)
        outputs, (hidden, cell) = self.rnn(src)

        outputs = self.maxpool(outputs).squeeze(0)

        outputs = self.fc(nn.ReLU()(self.li(outputs.squeeze(0))))

        return outputs


class QuestionLevelDifficulty_M(nn.Module):

    def __init__(self, args, tokenizer, roberta):
        super(QuestionLevelDifficulty_M, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args

        self.img_encoder = BackboneCNN(args)
        self.text_encoder = TextEncoder(args, roberta)

        self.memory_class = nn.Linear(768 * 4, 2)  # roberta hidden * memory level | 2, 3

        self.u_attn_layer = nn.ModuleList([TransformerAttn() for _ in range(2)])

        self.ham_attn = Attention()
        self.bilstm_u = Bilstm_OUR(args)
        self.bilstm_v = Bilstm_OUR(args)
        #
        self.li = nn.Linear(768 * 2, 768 * 2)
        self.u_li = nn.Linear(768, 768)
        self.v_li = nn.Linear(768, 768)

    def forward(self,
                text_inputs: Dict,
                img_inputs: Tensor,
                type: str) -> (Tensor, Tensor):

        u = self.img_encoder(img_inputs)
        v = self.text_encoder(text_inputs)

        attn_u = v
        attn_v = u

        for layer in self.u_attn_layer:
            attn_u = layer(attn_u, u, u, self.args)

        for layer in self.u_attn_layer:
            attn_v = layer(attn_v, v, v, self.args)

        uv = torch.cat([u, v], dim=1)
        uv = torch.cat([uv, attn_u], dim=1)
        uv = torch.cat([uv, attn_v], dim=1)

        memory_logits = self.memory_class(uv)

        return memory_logits, 'none', [1]


class QuestionLevelDifficulty_M_split(nn.Module):

    def __init__(self, args, tokenizer, roberta_que, roberta_uttr):
        super(QuestionLevelDifficulty_M_split, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args

        self.img_encoder = BackboneCNN(args)
        self.text_encoder_que = TextEncoder(args, roberta_que)
        self.text_encoder_uttr = TextEncoder(args, roberta_uttr)

        #self.bilstm_v = Bilstm_HAM(args)

        #self.ham_attn = Attention()

        self.memory_class = nn.Linear(768 * 9, 2)  # roberta hidden * memory level | 2, 3
        self.u_attn_layer = nn.ModuleList([TransformerAttn() for _ in range(2)])

    def forward(self,
                text_inputs: Dict,
                img_inputs: Tensor,
                type: str) -> (Tensor, Tensor):

        v = self.img_encoder(img_inputs)
        h = self.text_encoder_que(text_inputs[0])
        s = self.text_encoder_uttr(text_inputs[1])
        """
        # ham 구조
        v = self.ham_attn(v, v, v)
        h = self.ham_attn(h, h, h)
        s = self.ham_attn(s, s, s)

        v_h = self.ham_attn(v, h, h)
        v_s = self.ham_attn(v, s, s)

        s_h = self.ham_attn(s, h, h)
        s_v = self.ham_attn(s, v, v)

        v_tilde = torch.cat([v, v_h], dim=1)
        v_tilde = torch.cat([v_tilde, v_s], dim=1)

        s_tilde = torch.cat([s, s_h], dim=1)
        s_tilde = torch.cat([s_tilde, s_v], dim=1)

        v_result = self.bilstm_v(v_tilde)
        s_result = self.bilstm_v(s_tilde)

        result = v_result + s_result
        """
        attn_vh = h
        attn_vs = s
        for layer in self.u_attn_layer:
            attn_vh = layer(attn_vh, v, v, self.args)
        for layer in self.u_attn_layer:
            attn_vs = layer(attn_vs, v, v, self.args)

        attn_hs = s
        attn_hv = v
        for layer in self.u_attn_layer:
            attn_hs = layer(attn_hs, h, h, self.args)
        for layer in self.u_attn_layer:
            attn_hv = layer(attn_hv, h, h, self.args)

        attn_sv = v
        attn_sh = h
        for layer in self.u_attn_layer:
            attn_sv = layer(attn_sv, s, s, self.args)
        for layer in self.u_attn_layer:
            attn_sh = layer(attn_sh, s, s, self.args)

        result = torch.cat([v, h], dim=1)
        result = torch.cat([result, s], dim=1)
        result = torch.cat([result, attn_vh], dim=1)
        result = torch.cat([result, attn_vs], dim=1)

        result = torch.cat([result, attn_hv], dim=1)
        result = torch.cat([result, attn_hs], dim=1)

        result = torch.cat([result, attn_sv], dim=1)
        result = torch.cat([result, attn_sh], dim=1)

        result = self.memory_class(result)

        return result, 'none', 1


class TransformerAttn(nn.Module):
    def __init__(self):
        super(TransformerAttn, self).__init__()
        self.layer_norm = nn.LayerNorm(768)

        self.conv1 = nn.Conv1d(in_channels=768, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=768, kernel_size=1)

    def forward(self, query, key, value, args):

        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(768)# TVQA는 sqrt뺀다
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, value)

        context = self.scale_for_low_variance(context, args)

        output = nn.ReLU()(self.conv1(context.unsqueeze(1).transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2).squeeze(1)

        output = self.scale_for_low_variance(output, args)

        return output.squeeze(1)

    def scale_for_low_variance(self, value, args):
        maximum_value = torch.FloatTensor([math.sqrt(torch.max(value))]).to(args.device)
        if maximum_value > 1.0:
            value.divide_(maximum_value)
            return value
        return value


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value):

        scores = torch.matmul(query, key.transpose(-1, -2))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, value)

        return context