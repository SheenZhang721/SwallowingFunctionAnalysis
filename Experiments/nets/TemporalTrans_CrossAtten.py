from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import Experiments.Config as config


logger = logging.getLogger(__name__)


class Position_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,patchsize, img_size, n_channels, batch_size=config.batch_size):
        super().__init__()
        img_size = _pair(img_size)  # _pair is a function that returns a tuple of 2 elements
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14*14=196

        self.patch_embeddings = Conv2d(in_channels=n_channels,
                                       out_channels=n_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, batch_size))  # trainable parameter
        self.position_embeddings = nn.Parameter(torch.zeros(batch_size, 1, n_patches))  # trainable parameter
        self.dropout = Dropout(0.2)

    def forward(self, x):
        if x is None:  # x -> (B, C, H, W)
            return None
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)  # (B, C, H*W
        x = self.patch_embeddings(x)  # (B=8, C, n_patches^(1/2), n_patches^(1/2)) hidden = in_channels
        x = x.flatten(2)  # (B, C, n_patches) start_dim=2
        # x = x.transpose(-1, -2)  # (C, n_patches, B)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(B, T, C, -1)
        return embeddings


class TemporalRelativePosEmb(nn.Module):
    def __init__(self, num_frames, num_patches, embedding_dim=1):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.atten_score_dim = num_frames * num_patches

        # Relative position embedding
        self.relative_pos_range = 2 * num_frames - 1
        self.temporal_embedding = nn.Embedding(self.relative_pos_range, embedding_dim)

    def forward(self):
        # Compute relative positions
        frame_indices = torch.arange(self.num_frames, device=self.temporal_embedding.weight.device)
        frame_meshgrid = torch.meshgrid(frame_indices, frame_indices, indexing='ij')
        relative_pos = frame_meshgrid[0] - frame_meshgrid[1] + self.num_frames - 1

        # Embed relative positions
        pos_embeddings = self.temporal_embedding(relative_pos).squeeze()

        # Create block-wise bias map
        bias_map = torch.zeros(self.atten_score_dim, self.atten_score_dim,
                               device=pos_embeddings.device)
        for t1 in range(self.num_frames):
            for t2 in range(self.num_frames):
                block_start_row = t1 * self.num_patches
                block_start_col = t2 * self.num_patches

                bias_map[
                block_start_row:block_start_row + self.num_patches,
                block_start_col:block_start_col + self.num_patches
                ] = pos_embeddings[t1, t2]

        return bias_map

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, T, hidden, n_patch = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))  # sqrt(196) = 14
        # x = x.permute(0, 2, 1)  # (B, n_patch, hidden) -> (B, hidden, n_patch)
        x = x.contiguous().view(B*T, hidden, h, w)  # (B, hidden, n_patch) -> (B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x) # upsample is bilinear interpolation
        out = self.conv(x)  # 1x1 conv
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    '''
    Temporal Attention with Relative Position Bias
    '''
    def __init__(self, channel_dim, num_patches):
        super(Attention_org, self).__init__()
        self.num_channels = channel_dim
        self.num_frames = 8
        self.num_patches = num_patches
        self.KV_size = self.num_frames * self.num_patches  # 8*196
        # self.batch_size = channel_dim
        self.num_attention_heads = 4  # 4


        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):  # num_heads = 4
            query = nn.Linear(channel_dim, channel_dim, bias=False)  # linear transformation, Q
            key = nn.Linear( channel_dim,  channel_dim, bias=False)  # K
            value = nn.Linear(channel_dim,  channel_dim, bias=False)  # V
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(channel_dim, channel_dim, bias=False)
        self.attn_dropout = Dropout(0.2)
        self.proj_dropout = Dropout(0.2)

        # self.tempo_rel_pos_emb = TemporalRelativePosEmb(self.num_frames, self.num_patches)

        # coords = torch.stack(torch.meshgrid([torch.arange(self.num_frames), torch.arange(self.num_frames)]))
        # relative_coords = coords[0] - coords[1] + self.num_frames - 1
        # relative_coords = relative_coords.view(-1, self.num_frames)  # [B, B]  8*8
        # self.register_buffer('relative_position_index', relative_coords)  # register buffer to save it in the model, trainable=False
        # self.relative_position_table = nn.Parameter(
        #     torch.zeros(2 * self.num_frames - 1, self.num_attention_heads))  # trainable parameter  [15, 4*8*8]  [2b-1, num_heads * b * b]



    def forward(self, emb):
        B, T, C, N = emb.size()
        multi_head_Q_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb is not None:
            for query in self.query:
                Q = query(emb.permute(0, 1, 3, 2).contiguous().view(B, T*N, -1))  # (B, T, C, N) -> (B, T*N, C), Q
                multi_head_Q_list.append(Q)
        for key in self.key:
            K = key(emb.permute(0, 1, 3, 2).contiguous().view(B, T*N, -1))  # (B, T, C, N) -> (B, T*N, C)
            K = K.permute(0, 2, 1)  # (B, T*N, C) -> (B, C, T*N)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb.permute(0, 1, 3, 2).contiguous().view(B, T*N,-1))  # (B, T, C, N) -> (B, T*N, C), V
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q = torch.stack(multi_head_Q_list, dim=1) if emb is not None else None  # stack of 4 heads alone a new dimension
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        # multi_head_K = multi_head_K.transpose(-1, -2) if emb is not None else None  # (196, 4, B, C) -> (196, 4, C, B)

        attention_scores = torch.matmul(multi_head_Q, multi_head_K) if emb is not None else None  # (B, 4, T*N, C) * (B, 4, C, T*N) -> (B, 4, T*N, T*N)

        attention_scores = attention_scores / math.sqrt(self.KV_size) if emb is not None else None


        attention_scores = attention_scores.view(-1, self.num_attention_heads, self.KV_size, self.KV_size)  # without temporal relative position bias
        # attention_scores = attention_scores + self.tempo_rel_pos_emb()  # with temporal relative position bias

        attention_probs = self.softmax(self.psi(attention_scores)) if emb is not None else None  # attention_probs = softmax(attention_scores/ sqrt(d_k))
        # print(attention_probs4.size())

        attention_probs = self.attn_dropout(attention_probs) if emb is not None else None

        # multi_head_V = multi_head_V.transpose(-1, -2)  # (196, 4, C, B) -> (196, 4, B, C)
        context_layer = torch.matmul(attention_probs, multi_head_V) if emb is not None else None  # (B, 4, T*N, T*N) * (B, 4, T*N, C) -> (B, 4, T*N, C)

        context_layer = context_layer.permute(0, 3, 2, 1).contiguous() if emb is not None else None  # (B, 4, T*N, C) -> (B, C, T*N, 4), deep copy
        context_layer = context_layer.mean(dim=3) if emb is not None else None  # (B, C, T*N, 4) -> (B, C, T*N) mean along the head dim

        O1 = self.out(context_layer.permute(0, 2, 1)) if emb is not None else None  # (B, C, T*N) -> (B, T*N, C)
        O1 = O1.view(-1, self.num_frames, self.num_channels, self.num_patches)  # (B, T*N, C) -> (B, T, C, N)
        O1 = self.proj_dropout(O1) if emb is not None else None

        return O1




class Mlp(nn.Module):
    def __init__(self,in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    '''
    Transformer Block
    '''
    def __init__(self, n_patches=196, channel_dim=64):
        super(Block_ViT, self).__init__()
        expand_ratio = 4  # config sets the value to 4
        self.attn_norm = LayerNorm(n_patches,eps=1e-6)  # 8
        self.temporal_attn = Attention_org(channel_dim=channel_dim, num_patches=n_patches)
        self.ffn_norm = LayerNorm(n_patches,eps=1e-6)
        self.ffn = Mlp(n_patches,n_patches*expand_ratio)


    def forward(self, emb):
        # org = emb  # (196, B, C)
        org = emb   # (B, T, C, 196)
        # cx = self.attn_norm(emb.permute(0, 2, 1)) if emb is not None else None  # (196, B, C)
        # cx = cx.permute(0, 2, 1)
        cx = self.attn_norm(emb) if emb is not None else None  # layer norm
        cx = self.temporal_attn(cx)  # (B, C, 196) -> (B, C, 196)
        cx = org + cx if emb is not None else None  # (B, C, 196)

        org = cx
        # x = self.ffn_norm(cx.permute(0, 2, 1)) if emb is not None else None  # layer norm
        x = self.ffn_norm(cx)
        x = self.ffn(x) if emb is not None else None  # (196, B, C)

        # x = x.permute(0, 2, 1)
        x = x + org if emb is not None else None  # residual connection

        return x  # (B, C, 196)


class Encoder(nn.Module):
    '''
    Encoder is made up of self-attn and feed forward (defined below)
    '''
    def __init__(self, n_patches=196, channel_dim=64):

        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(n_patches,eps=1e-6)
        for _ in range(4):  # num_layers = 4
            layer = Block_ViT(n_patches, channel_dim=channel_dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb):
        # attn_weights = []
        for layer_block in self.layer:
            emb = layer_block(emb)  # (B, T, C, 196) -> (B, T, C, 196)
        emb = self.encoder_norm(emb) if emb is not None else None
        return emb


class TemporalTransformer_CrossAtten(nn.Module):
    def __init__(self, img_size, n_channels, patchSize=16):
        super().__init__()
        self.embeddings = Position_Embeddings(patchSize, img_size=img_size, n_channels=n_channels)
        self.encoder = Encoder(channel_dim=n_channels, n_patches = (img_size//patchSize)**2)
        self.reconstruct = Reconstruct(n_channels, n_channels, kernel_size=1,scale_factor=(patchSize, patchSize))
        self.num_frames = 8

    def forward(self,en):
        B, C, H, W = en.size()
        en_reshaped = en.view(B // self.num_frames, self.num_frames, C, H, W)  # [B, T, C, H, W]
        emb = self.embeddings(en_reshaped)  # (B, T, C, H, W) -> (B, T, C, 196)
        encoded = self.encoder(emb)  # (B, C, 196) -> (B, C, 196)
        x = self.reconstruct(encoded) if en is not None else None  # (B, C, 196) -> (B, C, 224, 224)

        x = x + en  if en is not None else None


        return x