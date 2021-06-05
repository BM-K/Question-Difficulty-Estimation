import os
import h5py
import torch
import logging
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score

logger = logging.getLogger(__name__)
writer = SummaryWriter()


def processing_images(args, info, mode=None):

    batch_image = []

    if args.dataset == 'MissO' or args.dataset == 'MissO_split':

        image_file = args.path_to_data + args.MissO_img_h5
        hq_file = h5py.File(image_file, 'r')

        for idx in range(len(info['qid'])):

            stack_images = hq_file[str(info['qid'][idx].strip())][:]

            stack_images = [img for step, img in enumerate(stack_images) if step % 3 == 0]

            batch_image.append(torch.tensor(stack_images).to(args.device))

    else:

        image_file = args.path_to_data + args.TVQA_img_h5
        hq_file = h5py.File(image_file, 'r')

        for idx in range(len(info)):

            stack_images = hq_file[str(info[idx].strip())][:]

            stack_images = [img for step, img in enumerate(stack_images) if step % 3 == 0]

            batch_image.append(torch.tensor(stack_images).to(args.device))

    hq_file.close()

    return batch_image


def cal_acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float().mean()

        f1 = f1_score(y.cpu(), yhat.cpu())#, average='macro')

        re = recall_score(y.cpu(), yhat.cpu())
        pr = precision_score(y.cpu(), yhat.cpu())

    return acc, f1, re, pr


def get_bert_parameter(inputs, pad_token_idx, sep_token_idx, args):
    sep_index = [idx for idx, value in enumerate(inputs) if value == sep_token_idx]

    if len(sep_index) == 0:
        segment_ids = [0] * args.max_len
    else:
        segment_ids = ([0] * (sep_index[0] + 1)) + ([1] * (args.max_len - sep_index[0] - 1))

    pad_index = [idx for idx, value in enumerate(inputs) if value == pad_token_idx]
    if len(pad_index) == 0:
        attention_mask = [1] * args.max_len
    else:
        attention_mask = ([1] * pad_index[0]) + ([0] * (args.max_len - pad_index[0]))

    segment_ids = torch.tensor(segment_ids)
    attention_mask = torch.tensor(attention_mask)

    return segment_ids, attention_mask.float()


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']


def epoch_time(start_time, end_time) -> (int, int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def save_model(config, cp, pco, pp):
    """
    cp (current performance) has train valid loss and train valid perplexity
    pco (performance_check_objects)
    saved model's name | epoch-{}-loss-{}.pt | in args.path_to_save
    """
    if not os.path.exists(config['args'].path_to_save):
        os.makedirs(config['args'].path_to_save)

    sorted_path = config['args'].path_to_save + '/checkpoint-epoch-{}-loss-{}.pt'.format(str(cp['ep']), round(cp['vl'], 4))

    writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])
    writer.add_scalars('memory_acc_graph', {'train': cp['tma'], 'valid': cp['vma']}, cp['ep'])
    writer.add_scalars('logic_acc_graph', {'train': cp['tla'], 'valid': cp['vla']}, cp['ep'])

    if cp['ep'] + 1 == config['args'].epochs:
        writer.close()

    if cp['vl'] < pco['best_valid_loss'][0]:
    #if cp['vma'] > pco['best_valid_acc'][0]:
        pco['early_stop_check'] = [0]
        pco['best_valid_loss'][0] = cp['vl']
        #pco['best_valid_acc'][0] = cp['vma']
        #state = {
        #        'State_dict': config['model'].state_dict(),
        #        'optimizer': config['optimizer'].state_dict()
        #}
        torch.save(config['model'].state_dict(), sorted_path)
        #torch.save(state, sorted_path)
        print(f'\n\t## SAVE valid_loss: {cp["vl"]:.4f} | valid memory acc: {cp["vma"]:.4f} | valid logic acc: {cp["vla"]:.4f} ##')

    print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
    print(f'\t==Train Loss: {cp["tl"]:.4f} | Train memory acc: {cp["tma"]:.4f} | Train logic acc: {cp["tla"]:.4f}==')
    print(f'\t==Valid Loss: {cp["vl"]:.4f} | Valid memory acc: {cp["vma"]:.4f} | Valid logic acc: {cp["vla"]:.4f}==')
    print(f'\t==Epoch latest LR: {get_lr(config["optimizer"]):.9f}==\n')

    return pco['early_stop_check'], pco['best_valid_loss']
