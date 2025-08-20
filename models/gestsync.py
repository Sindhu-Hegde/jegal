import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class GestSync(nn.Module):

    def __init__(self):
        super().__init__()

        self.net_vid = self.build_net_vid()
        self.ff_vid = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1024)
            )

        self.pos_encoder = PositionalEncoding_GestSync(d_model=512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.net_aud = self.build_net_aud()
        self.lstm = nn.LSTM(512, 256, num_layers=1, bidirectional=True, batch_first=True)

        self.ff_aud = NetFC_2D(input_dim=512, hidden_dim=512, embed_dim=1024)


        self.logits_scale = nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.logits_scale.weight)

        self.fc = nn.Linear(1,1)

    def build_net_vid(self):
        layers = [
            {
                'type': 'conv3d',
                'n_channels': 64,
                'kernel_size': (5, 7, 7),
                'stride': (1, 3, 3),
                'padding': (0),
                'maxpool': {
                    'kernel_size': (1, 3, 3),
                    'stride': (1, 2, 2)
                }
            },
            {
                'type': 'conv3d',
                'n_channels': 128,
                'kernel_size': (1, 5, 5),
                'stride': (1, 2, 2),
                'padding': (0, 0, 0),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 2, 2),
                'padding': (0, 1, 1),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 2),
                'padding': (0, 1, 1),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 1),
                'padding': (0, 1, 1),
                'maxpool': {
                    'kernel_size': (1, 3, 3),
                    'stride': (1, 2, 2)
                }
            },
            {
                'type': 'fc3d',
                'n_channels': 512,
                'kernel_size': (1, 4, 4),
                'stride': (1, 1, 1),
                'padding': (0),
            },
        ]
        return VGGNet(n_channels_in=3, layers=layers)

    def build_net_aud(self):
        layers = [
            {
                'type': 'conv2d',
                'n_channels': 64,
                'kernel_size': (3, 3),
                'stride': (2, 2),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (3, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'conv2d',
                'n_channels': 192,
                'kernel_size': (3, 3),
                'stride': (1, 2),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (3, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'conv2d',
                'n_channels': 384,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
            },
            {
                'type': 'conv2d',
                'n_channels': 256,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
            },
            {
                'type': 'conv2d',
                'n_channels': 256,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (2, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'fc2d',
                'n_channels': 512,
                'kernel_size': (4, 2),
                'stride': (1, 1),
                'padding': (0, 0),
            },
        ]
        return VGGNet(n_channels_in=1, layers=layers)

    def forward_vid(self, x, return_feats=False):
        out_conv = self.net_vid(x).squeeze(-1).squeeze(-1)
        # print("Conv: ", out_conv.shape)                          # Bx1024x21x1x1

        out = self.pos_encoder(out_conv.transpose(1,2))
        out_trans = self.transformer_encoder(out)
        # print("Transformer: ", out_trans.shape)                   # Bx21x1024

        out = self.ff_vid(out_trans).transpose(1,2)
        # print("MLP output: ", out.shape)                          # Bx1024

        if return_feats:
            return out, out_conv
        else:
            return out

    def forward_aud(self, x):
        out = self.net_aud(x)
        out = self.ff_aud(out)
        out = out.squeeze(-1)
        return out


class PositionalEncoding_GestSync(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding_GestSync, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class DebugModule(nn.Module):
    """
    Wrapper class for printing the activation dimensions 
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.debug_log = True

    def debug_line(self, layer_str, output, memuse=1, final_call=False):
        if self.debug_log:
            namestr = '{}: '.format(self.name) if self.name is not None else ''
            # print('{}{:80s}: dims {}'.format(namestr, repr(layer_str),
            #                                  output.shape))

            if final_call:
                self.debug_log = False
                # print()

def calc_receptive_field(layers, imsize, layer_names=None):
    if layer_names is not None:
        print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]

    for l_id, layer in enumerate(layers):
        conv = [
            layer[key][-1] if type(layer[key]) in [list, tuple] else layer[key]
            for key in ['kernel_size', 'stride', 'padding']
        ]
        currentLayer = outFromIn(conv, currentLayer)
        if 'maxpool' in layer:
            conv = [
                (layer['maxpool'][key][-1] if type(layer['maxpool'][key])
                 in [list, tuple] else layer['maxpool'][key]) if
                (not key == 'padding' or 'padding' in layer['maxpool']) else 0
                for key in ['kernel_size', 'stride', 'padding']
            ]
            currentLayer = outFromIn(conv, currentLayer, ceil_mode=False)
    return currentLayer

def outFromIn(conv, layerIn, ceil_mode=True):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out
    
class VGGNet(DebugModule):

    conv_dict = {
        'conv1d': nn.Conv1d,
        'conv2d': nn.Conv2d,
        'conv3d': nn.Conv3d,
        'fc1d': nn.Conv1d,
        'fc2d': nn.Conv2d,
        'fc3d': nn.Conv3d,
    }

    pool_dict = {
        'conv1d': nn.MaxPool1d,
        'conv2d': nn.MaxPool2d,
        'conv3d': nn.MaxPool3d,
    }

    norm_dict = {
        'conv1d': nn.BatchNorm1d,
        'conv2d': nn.BatchNorm2d,
        'conv3d': nn.BatchNorm3d,
        'fc1d': nn.BatchNorm1d,
        'fc2d': nn.BatchNorm2d,
        'fc3d': nn.BatchNorm3d,
    }

    def __init__(self, n_channels_in, layers):
        super(VGGNet, self).__init__()

        self.layers = layers

        n_channels_prev = n_channels_in
        for l_id, lr in enumerate(self.layers):
            l_id += 1
            name = 'fc' if 'fc' in lr['type'] else 'conv'
            conv_type = self.conv_dict[lr['type']]
            norm_type = self.norm_dict[lr['type']]
            self.__setattr__(
                '{:s}{:d}'.format(name, l_id),
                conv_type(n_channels_prev,
                          lr['n_channels'],
                          kernel_size=lr['kernel_size'],
                          stride=lr['stride'],
                          padding=lr['padding']))
            n_channels_prev = lr['n_channels']
            self.__setattr__('bn{:d}'.format(l_id), norm_type(lr['n_channels']))
            if 'maxpool' in lr:
                pool_type = self.pool_dict[lr['type']]
                padding = lr['maxpool']['padding'] if 'padding' in lr[
                    'maxpool'] else 0
                self.__setattr__(
                    'mp{:d}'.format(l_id),
                    pool_type(kernel_size=lr['maxpool']['kernel_size'],
                              stride=lr['maxpool']['stride'],
                              padding=padding),
                )

    def forward(self, inp):
        self.debug_line('Input', inp)
        out = inp
        for l_id, lr in enumerate(self.layers):
            l_id += 1
            name = 'fc' if 'fc' in lr['type'] else 'conv'
            out = self.__getattr__('{:s}{:d}'.format(name, l_id))(out)
            out = self.__getattr__('bn{:d}'.format(l_id))(out)
            out = nn.ReLU(inplace=True)(out)
            self.debug_line(self.__getattr__('{:s}{:d}'.format(name, l_id)),
                            out)
            if 'maxpool' in lr:
                out = self.__getattr__('mp{:d}'.format(l_id))(out)
                self.debug_line(self.__getattr__('mp{:d}'.format(l_id)), out)

        self.debug_line('Output', out, final_call=True)

        return out



class NetFC(DebugModule):

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NetFC, self).__init__()
        self.fc7 = nn.Conv3d(input_dim, hidden_dim, kernel_size=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(hidden_dim)
        self.fc8 = nn.Conv3d(hidden_dim, embed_dim, kernel_size=(1, 1, 1))

    def forward(self, inp):
        out = self.fc7(inp)
        self.debug_line(self.fc7, out)
        out = self.bn7(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc8(out)
        self.debug_line(self.fc8, out, final_call=True)
        return out

class NetFC_2D(DebugModule):

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NetFC_2D, self).__init__()
        self.fc7 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.bn7 = nn.BatchNorm2d(hidden_dim)
        self.fc8 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=(1, 1))

    def forward(self, inp):
        out = self.fc7(inp)
        self.debug_line(self.fc7, out)
        out = self.bn7(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc8(out)
        self.debug_line(self.fc8, out, final_call=True)
        return out