import logging
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_loader
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig, BertModel, BertConfig
from models.QLD.qld_memory import QuestionLevelDifficulty_M, QuestionLevelDifficulty_M_split
from models.QLD.qld_logic import QuestionLevelDifficulty_L
from models.QLD.qld_only_text import QuestionLevelDifficultyOT
logger = logging.getLogger(__name__)


def get_loss_func(tokenizer):
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optim(args, model) -> optim:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    return optimizer


def get_scheduler(optim, args, train_loader) -> get_linear_schedule_with_warmup:
    train_total = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_step, num_training_steps=train_total)
    return scheduler


def processing_model(config, text_inputs, image_inputs, type=None):

    if config['args'].eval == 'False':
        if config['args'].only_text_input == 'True':
            memory_logits, logic_logits, pp = config['model'](text_inputs, type)
        else:
            memory_logits, logic_logits, pp = config['model'](text_inputs, image_inputs, type)
    else:

        if config['args'].only_text_input == 'True':
            memory_logits, logic_logits, pp = config['model'](text_inputs, type)
        else:
            memory_logits, _, pp = config['memory_model'](text_inputs, image_inputs, type)
            logic_logits = config['logic_model'](text_inputs, image_inputs)

    return memory_logits, logic_logits, pp


def model_setting(args):
    loader, tokenizer = get_loader(args)

    if args.text_processor == 'roberta':
        config = RobertaConfig()
        roberta = RobertaModel(config)
        # text_processor = roberta.from_pretrained('roberta-base')
        ## 텍스트를 분할해서 로벌타에 넣어보자
        if args.dataset == 'MissO_split' or args.dataset == 'TVQA_split':
            text_processor_que = roberta.from_pretrained('roberta-base')
            text_processor_utt = roberta.from_pretrained('roberta-base')
        elif args.eval == 'True':
            memory_processor = roberta.from_pretrained('roberta-base')
            logic_processor = roberta.from_pretrained('roberta-base')
        else:
            text_processor = roberta.from_pretrained('roberta-base')

    elif args.text_processor == 'bert':
        config = BertConfig()
        bert = BertModel(config)
        text_processor = bert.from_pretrained('bert-base-uncased')
    else:
        text_processor = None

    if args.eval == 'False':
        if args.only_text_input == 'True':
            model = QuestionLevelDifficultyOT(args, tokenizer, text_processor)
        else:
            if args.dataset == 'MissO_split' or args.dataset == 'TVQA_split':
                model = QuestionLevelDifficulty_M_split(args, tokenizer, text_processor_que, text_processor_utt)
            else:
                model = QuestionLevelDifficulty_M(args, tokenizer, text_processor)

        criterion = get_loss_func(tokenizer)
        optimizer = get_optim(args, model)
        scheduler = get_scheduler(optimizer, args, loader['train'])

        model.to(args.device)
        criterion.to(args.device)

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': args,
                  'model': model}
    else:
        memory_model = QuestionLevelDifficulty_M(args, tokenizer, memory_processor)
        logic_model = QuestionLevelDifficulty_L(args, tokenizer, logic_processor)

        memory_model.to(args.device)
        logic_model.to(args.device)

        config = {'loader': loader,
                  'tokenizer': tokenizer,
                  'args': args,
                  'memory_model': memory_model,
                  'logic_model': logic_model}

    return config