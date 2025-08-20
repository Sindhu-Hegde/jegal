import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class Encoder_Transformer(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, final_norm=True):
        super(Encoder_Transformer, self).__init__()
        self.layers = clones(layer, N)
        if final_norm: self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return (self.norm(x) if hasattr(self, 'norm') else x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer_Transformer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer_Transformer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print("Scores: ", scores.shape)
    # with autocast(enabled=False):
    if mask is not None:
        scores = scores.float()
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MultiHeadedAttention_Transformer(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_Transformer, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print("Mask inside attn: ", mask.shape)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward_Transformer(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward_Transformer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding_Transformer(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding_Transformer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



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