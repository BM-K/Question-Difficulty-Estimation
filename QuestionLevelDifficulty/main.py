import time
import torch
from tqdm import tqdm
from models.QLD.functions import processing_model, model_setting
from models.setting import set_args, set_logger, set_seed, print_args
from models.utils import save_model, get_lr, epoch_time, cal_acc, processing_images


def system_setting():
    args = set_args()
    print_args(args)
    set_seed(args)

    early_stop_check = [0]
    best_valid_loss = [float('inf')]
    performance_check_objects = {'early_stop_check': early_stop_check,
                                 'best_valid_loss': best_valid_loss}

    return args, performance_check_objects


def train(config) -> (float, float, float):
    total_loss = 0
    iter_num = 0
    train_mem_acc = 0
    train_log_acc = 0

    logger.info("Training main")
    for step, batch in enumerate(tqdm(config['loader']['train'])):
        config['optimizer'].zero_grad()

        input_sentence, img_info, memory, logic = batch

        if config['args'].only_text_input == 'True':
            img_data = None
        else:
            img_data = processing_images(config['args'], img_info)

        memory_logits, logic_logits = processing_model(config, input_sentence, img_data)
        memory_loss = config['criterion'](memory_logits, memory)
        #logic_loss = config['criterion'](logic_logits, logic)

        loss = memory_loss# + logic_loss
        loss.backward()

        config['optimizer'].step()
        config['scheduler'].step()

        total_loss += loss
        iter_num += 1

        with torch.no_grad():
            tr_mem_acc = cal_acc(memory_logits, memory)
            #tr_log_acc = cal_acc(logic_logits, logic)

        train_mem_acc += tr_mem_acc
        #train_log_acc += tr_log_acc

    return total_loss.data.cpu().numpy() / iter_num,\
           train_mem_acc.data.cpu().numpy() / iter_num,\
           #train_log_acc.data.cpu().numpy() / iter_num


def valid(config) -> (float, float, float):
    total_loss = 0
    iter_num = 0
    valid_mem_acc = 0
    valid_log_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['valid']):

            input_sentence, img_info, memory, logic = batch

            if config['args'].only_text_input == 'True':
                img_data = None
            else:
                img_data = processing_images(config['args'], img_info)

            memory_logits, logic_logits = processing_model(config, input_sentence, img_data)
            memory_loss = config['criterion'](memory_logits, memory)
            #logic_loss = config['criterion'](logic_logits, logic)

            loss = memory_loss# + logic_loss

            total_loss += loss
            iter_num += 1

            with torch.no_grad():
                tr_mem_acc = cal_acc(memory_logits, memory)
                #tr_log_acc = cal_acc(logic_logits, logic)

            valid_mem_acc += tr_mem_acc
            #valid_log_acc += tr_log_acc

    return total_loss.data.cpu().numpy() / iter_num,\
           valid_mem_acc.cpu().numpy() / iter_num,\
           #valid_log_acc.cpu().numpy() / iter_num


def test(config) -> (float, float, float):
    total_loss = 0
    iter_num = 0
    test_mem_acc = 0
    test_log_acc = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['test']):

            input_sentence, img_info, memory, logic = batch

            if config['args'].only_text_input == 'True':
                img_data = None
            else:
                img_data = processing_images(config['args'], img_info)

            memory_logits, logic_logits = processing_model(config, input_sentence, img_data)
            memory_loss = config['criterion'](memory_logits, memory)
            #logic_loss = config['criterion'](logic_logits, logic)

            loss = memory_loss# + logic_loss

            total_loss += loss
            iter_num += 1

            with torch.no_grad():
                tr_mem_acc = cal_acc(memory_logits, memory)
                #tr_log_acc = cal_acc(logic_logits, logic)

            test_mem_acc += tr_mem_acc
            #test_log_acc += tr_log_acc

    return total_loss.data.cpu().numpy() / iter_num, \
           test_mem_acc.cpu().numpy() / iter_num, \
           #test_log_acc.cpu().numpy() / iter_num


def main() -> None:
    """
    config is made up of
    dictionary {data loader, optimizer, criterion, scheduler, tokenizer, args, model}
    """
    args, performance_check_objects = system_setting()
    config = model_setting(args)

    if args.train_ == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):
            start_time = time.time()

            config['model'].train()
            train_loss, train_mem_acc,  = train(config)

            config['model'].eval()
            valid_loss, valid_mem_acc,  = valid(config)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            performance = {'tl': train_loss, 'vl': valid_loss, 'tma': train_mem_acc, 'tla': 0,
                           'vma': valid_mem_acc, 'vla': 0,
                           'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

            performance_check_objects['early_stop_check'], performance_check_objects['best_valid_loss'] = \
                save_model(config, performance, performance_check_objects)

    if args.test_ == 'True':
        logger.info("Start Test")

        config['model'].load_state_dict(torch.load(args.path_to_saved_model))
        config['model'].eval()

        test_loss, test_mem_acc,  = test(config)
        print(f'\n\t==Test loss: {test_loss:.3f} | Test memory acc: {test_mem_acc:.3f} | Test logic acc: {0.1:.3f}==\n')


if __name__ == '__main__':
    logger = set_logger()
    main()