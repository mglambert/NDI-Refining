from xQSM import xQSM
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import time

def zero_padding(array, factor=8):
    original_shape = np.array(array.shape)

    new_shape = np.ceil(original_shape / factor) * factor
    new_shape = new_shape.astype(int)

    padding_width = new_shape - original_shape
    pad_before = (padding_width // 2).astype(int)
    pad_after = padding_width - pad_before

    pad_config = [(pad_before[i], pad_after[i]) for i in range(len(original_shape))]

    padded_array = np.pad(array, pad_config, mode='constant', constant_values=0)

    padding_info = {
        'original_shape': original_shape,
        'pad_width': pad_config
    }

    return padded_array, padding_info


def zero_removing(padded_array, padding_info):
    original_shape = padding_info['original_shape']
    pad_width = padding_info['pad_width']

    slices = tuple(slice(pad_width[i][0], pad_width[i][0] + original_shape[i])
                   for i in range(len(original_shape)))

    unpadded_array = padded_array[slices]

    return unpadded_array


def eval(Field):
    print('Padding')
    Field, padding_info = zero_padding(Field)
    Field = torch.from_numpy(Field)
    Field = torch.unsqueeze(Field, 0)
    Field = torch.unsqueeze(Field, 0)
    Field = Field.float()

    print('Load Pretrained Network')
    model_weights_path = 'xQSM_invivo_withNoiseLayer.pth'
    Net = xQSM(2)
    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device("cuda:0")
        print(torch.cuda.get_device_name(0))
        Net.to(device)
        Net = nn.DataParallel(Net)
        Net.load_state_dict(torch.load(model_weights_path))
        Net = Net.module
        Net.to(device)
        Net.eval()
        Field = Field.to(device)
    else:
        weights = torch.load(model_weights_path, map_location='cpu')
        new_state_dict = OrderedDict()
        print(new_state_dict)
        for k, v in weights.items():
            name = k[7:]
            new_state_dict[name] = v
        Net.load_state_dict(new_state_dict)
        Net.eval()
    ################ Evaluation ##################
    print('Reconstruction')

    with torch.no_grad():
        time_start = time.time()
        Recon = Net(Field)
        time_end = time.time()
        print('%f seconds elapsed!' % (time_end - time_start))
        Recon = torch.squeeze(Recon, 0)
        Recon = torch.squeeze(Recon, 0)
        Recon = Recon.to('cpu')
        Recon = Recon.numpy()

    Recon = zero_removing(Recon, padding_info)
    return Recon

