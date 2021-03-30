import os
import math
import numpy
import torch
import logging
from tensorboardX import SummaryWriter
from torchvision import transforms
from PIL import Image
from glob import glob
from skimage import io

logger = logging.getLogger(__name__)
writer = SummaryWriter()
image_size = [224, 224]
images_trans = transforms.Compose([transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])


def processing_images(args, info):
    # vid : AnotherMissOh01_025_0889
    batch_image = []

    for idx in range(len(info['vid'])):
        vid = info['vid'][idx]
        contained = info['shot'][idx]

        images_tensor = []
        shot_contained = []
        episode = vid.split('_')[0][-2:]
        scene, shot = vid.split('_')[1], vid.split('_')[2]
        contained = contained.split(',')

        if len(contained) == 2:
            a, b = contained[0].replace('[', '').strip(), contained[1].replace(']', '').strip()
            for i in range(int(a), int(b) + 1):
                if len(str(i)) == 1:
                    shot_contained.append('000' + str(i))
                elif len(str(i)) == 2:
                    shot_contained.append('00' + str(i))
                elif len(str(i)) == 3:
                    shot_contained.append('0' + str(i))
                elif len(str(i)) == 4:
                    shot_contained.append(str(i))
                else:
                    exit()
        else:
            shot_contained = contained[0].replace('[', '').replace(']', '')
            if len(str(shot_contained)) == 1:
                shot_contained = '000' + str(shot_contained)
            elif len(str(shot_contained)) == 2:
                shot_contained = '00' + str(shot_contained)
            elif len(str(shot_contained)) == 3:
                shot_contained = '0' + str(shot_contained)
            elif len(str(shot_contained)) == 4:
                shot_contained = str(shot_contained)
            else:
                exit()
            shot_contained = [str(shot_contained)]

        iter_check = 0
        for i in range(len(shot_contained)):
            if iter_check == 100:break

            temp_path = args.path_to_image \
                        + str(episode) + '/' + str(scene) + '/' + str(shot_contained[i]) + '/*.jpg'

            temp_path = sorted(glob(temp_path))
            
            for image_path in temp_path:
                if iter_check == 100:break

                image = images_trans(Image.fromarray(io.imread(image_path)))
                image = image.to(torch.float32)
                images_tensor.append(image)
                iter_check += 1

        batch_image.append(images_tensor)
    
    return batch_image


def cal_acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float().mean()
    return acc


def get_bert_parameter(inputs, pad_token_idx, sep_token_idx, args):
    sep_index = [idx for idx, value in enumerate(inputs) if value == sep_token_idx]
    segment_ids = ([0] * (sep_index[0] + 1)) + ([1] * (args.max_len - sep_index[0] - 1))

    pad_index = [idx for idx, value in enumerate(inputs) if value == pad_token_idx]
    if len(pad_index) == 0:
        attention_mask = [1] * args.max_len
    else:
        attention_mask = ([1] * (pad_index[0] + 1)) + ([0] * (args.max_len - pad_index[0] - 1))


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


def save_model(config, cp, pco):
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
        pco['early_stop_check'] = [0]
        pco['best_valid_loss'][0] = cp['vl']
        torch.save(config['model'].state_dict(), sorted_path)
        print(f'\n\t## SAVE valid_loss: {cp["vl"]:.3f} | valid memory acc: {cp["vma"]:.3f} | valid logic acc: {cp["vla"]:.3f} ##')

    print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
    print(f'\t==Train Loss: {cp["tl"]:.3f} | Train memory acc: {cp["tma"]:.3f} | Train logic acc: {cp["tla"]:.3f}==')
    print(f'\t==Valid Loss: {cp["vl"]:.3f} | Valid memory acc: {cp["vma"]:.3f} | Valid logic acc: {cp["vla"]:.3f}==')
    print(f'\t==Epoch latest LR: {get_lr(config["optimizer"]):.9f}==\n')

    return pco['early_stop_check'], pco['best_valid_loss']
