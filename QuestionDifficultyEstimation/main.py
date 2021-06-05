import csv
import time
import torch
from tqdm import tqdm
from models.QLD.functions import processing_model, model_setting
from models.setting import set_args, set_logger, set_seed, print_args
from models.utils import save_model, get_lr, epoch_time, cal_acc, processing_images, print_size_of_model
from tensorboardX import SummaryWriter

writer = SummaryWriter()


def system_setting():
    args = set_args()
    print_args(args)
    set_seed(args)

    early_stop_check = [0]
    best_valid_loss = [float('inf')]
    #best_valid_loss = [float('-inf')]
    performance_check_objects = {'early_stop_check': early_stop_check,
                                 'best_valid_loss': best_valid_loss}

    return args, performance_check_objects


def train(config, global_steps) -> (float, float, float):
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
            img_data = processing_images(config['args'], img_info, mode='train')

        memory_logits, _, pp = processing_model(config, input_sentence, img_data, type='train')
        memory_loss = config['criterion'](memory_logits, memory)

        loss = memory_loss
        loss.backward()

        config['optimizer'].step()
        config['scheduler'].step()

        total_loss += loss
        iter_num += 1

        with torch.no_grad():
            tr_mem_acc, _, _, _ = cal_acc(memory_logits, memory)

        train_mem_acc += tr_mem_acc

    return total_loss.data.cpu().numpy() / iter_num,\
           train_mem_acc.data.cpu().numpy() / iter_num, \
           pp


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
                img_data = processing_images(config['args'], img_info, mode='valid')

            memory_logits, _, _ = processing_model(config, input_sentence, img_data)
            memory_loss = config['criterion'](memory_logits, memory)

            loss = memory_loss

            total_loss += loss
            iter_num += 1

            with torch.no_grad():
                tr_mem_acc, _,_,_ = cal_acc(memory_logits, memory)

            valid_mem_acc += tr_mem_acc

    return total_loss.data.cpu().numpy() / iter_num,\
           valid_mem_acc.cpu().numpy() / iter_num,\


def test(config, correct_list) -> (float, float, float):
    total_loss = 0
    iter_num = 0
    test_mem_acc = 0
    test_f1_score = 0
    total_step = 0
    test_re_score = 0
    test_pr_score = 0

    y_tor = torch.tensor([])
    yhat_tor = torch.tensor([])

    from sklearn.metrics import f1_score, recall_score, precision_score

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['test']):

            input_sentence, img_info, memory, logic = batch

            if config['args'].only_text_input == 'True':
                img_data = None
            else:
                img_data = processing_images(config['args'], img_info, mode='test')

            memory_logits, _, _ = processing_model(config, input_sentence, img_data)
            memory_loss = config['criterion'](memory_logits, memory)

            loss = memory_loss

            total_loss += loss
            iter_num += 1

            with torch.no_grad():
                tr_mem_acc, _, re, pr = cal_acc(memory_logits, memory)

                y_tor = torch.cat([y_tor, memory.cpu()], dim=-1)

                yhat = memory_logits.max(dim=-1)[1]

                yhat_tor = torch.cat([yhat_tor, yhat.cpu()], dim=-1)

                corr = yhat == memory
                for i in range(len(memory)):
                    correct_list.append([str(memory[i].cpu().numpy()), str(corr[i].cpu().numpy())])

            test_mem_acc += tr_mem_acc
            total_step += 1

    with open('correct_list.tsv', 'w', encoding='utf-8') as f:
        wt = csv.writer(f, delimiter='\t')
        for i in range(len(correct_list)):
            wt.writerow(correct_list[i])

    #f1 = f1_score(y_tor, yhat_tor)
    re = recall_score(y_tor, yhat_tor)#, average='macro')
    pr = precision_score(y_tor, yhat_tor)#, average='macro')
    f1 = 2*((re*pr)/(re+pr))

    print(f"\n{test_mem_acc.cpu().numpy() / iter_num}")
    print(f"f1: {f1}")
    print(f"recall: {re}")
    print(f"precision: {pr}")

    return total_loss.data.cpu().numpy() / iter_num, \
           test_mem_acc.cpu().numpy() / iter_num, \
           test_f1_score / iter_num


def inference(config):

    iter_num = 0
    memory_acc = 0
    memory_f1_score = 0
    logic_acc = 0
    logic_f1_score = 0

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['test']):
            
            input_sentence, img_info, memory, logic = batch

            if config['args'].only_text_input == 'True':
                img_data = None
            else:
                img_data = processing_images(config['args'], img_info, mode='test')

            memory_logits, logic_logits, _ = processing_model(config, input_sentence, img_data)

            iter_num += 1

            with torch.no_grad():
                mem_acc, mem_f1_score = cal_acc(memory_logits, memory)
                log_acc, log_f1_score = cal_acc(logic_logits, logic)

            memory_acc += mem_acc
            memory_f1_score += mem_f1_score
            logic_acc += log_acc
            logic_f1_score += log_f1_score

    print(f'\t==Memory acc: {memory_acc.cpu().numpy() / iter_num:.4f} | Memory f1: {memory_f1_score / iter_num:.4f}==')
    print(f'\t==Logic acc: {logic_acc.cpu().numpy() / iter_num:.4f} | Logic f1: {logic_f1_score / iter_num:.4f}==')
    exit()
    """
    # 1 batch로 바꿔야함
    total_step = 0

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(config['loader']['test']):

            input_sentence, img_info, memory, logic = batch
            img_data = processing_images(config['args'], img_info, mode='test')
            memory_logits, logic_logits, _ = processing_model(config, input_sentence, img_data)
            
            print(f"Question: {config['tokenizer'].convert_ids_to_tokens(input_sentence['input_ids'].squeeze(0))}")
            mem = memory_logits.max(dim=-1)[1]
            lo = logic_logits.max(dim=-1)[1]
            print(f"Memory level: {mem + 2}")
            print(f"Logic level: {lo + 1}")
            exit()
            
            total_step += 1

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    total_time = epoch_mins * 60 + epoch_secs
    print(f"Average Time: {total_time}")
    print(f"Average Time: {total_time/total_step}")
    exit()
    """

def main() -> None:
    """
    config is made up of
    dictionary {data loader, optimizer, criterion, scheduler, tokenizer, args, model}
    """
    args, performance_check_objects = system_setting()
    config = model_setting(args)
    global_steps = [0]

    if args.train_ == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):
            start_time = time.time()

            config['model'].train()
            train_loss, train_mem_acc, pp = train(config, global_steps)

            config['model'].eval()
            valid_loss, valid_mem_acc,  = valid(config)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            performance = {'tl': train_loss, 'vl': valid_loss, 'tma': train_mem_acc, 'tla': 0,
                           'vma': valid_mem_acc, 'vla': 0,
                           'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

            performance_check_objects['early_stop_check'], performance_check_objects['best_valid_loss'] = \
                save_model(config, performance, performance_check_objects, pp)

    correct_list = []
    if args.test_ == 'True':
        logger.info("Start Test")

        config['model'].load_state_dict(torch.load(args.path_to_saved_model))#, strict=False)
        #config['model'].load_state_dict(torch.load(args.path_to_saved_model)['State_dict'])
        config['model'].eval()

        test_loss, test_mem_acc, f1 = test(config, correct_list)
        print(f'\n\t==Test loss: {test_loss:.4f} | Test memory acc: {test_mem_acc:.4f} | Test logic acc: {0.1:.4f} | Test F1 score: {f1:.4f}==\n')

    if args.eval == 'True':
        logger.info("Start Inference")

        config['memory_model'].load_state_dict(torch.load('./output/memory_our_9805.pt'), strict=False)
        config['logic_model'].load_state_dict(torch.load('./output/logic_8640.pt'), strict=False)

        print("=========================================================")
        print(f"Memory model size: ")
        print_size_of_model(config['memory_model'])
        print(f"Logic model size: ")
        print_size_of_model(config['logic_model'])
        print("=========================================================")

        config['memory_model'].eval()
        config['logic_model'].eval()

        inference(config)


if __name__ == '__main__':
    logger = set_logger()
    main()