import logging
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_loader
from transformers import get_linear_schedule_with_warmup, RobertaModel, RobertaConfig
from models.QLD.qld import QuestionLevelDifficulty
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


def processing_model(config, text_inputs, image_inputs):
    memory_logits, logic_logits = config['model'](text_inputs, image_inputs)

    return memory_logits, logic_logits


def model_setting(args):
    loader, tokenizer = get_loader(args)

    config = RobertaConfig()
    roberta = RobertaModel(config)
    roberta = roberta.from_pretrained('roberta-base')

    if args.only_text_input == 'True':
        model = QuestionLevelDifficultyOT(args, tokenizer, roberta)
    else:
        model = QuestionLevelDifficulty(args, tokenizer, roberta)

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

    return config