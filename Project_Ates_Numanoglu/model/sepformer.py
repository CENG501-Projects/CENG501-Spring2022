"""

    This code consists of the reconstruction of the SepFormer model based on the methods 
    mentioned in SFSRNet: Super-resolution for single-channel Audio Source Separation.

    The base code is taken from public GitHup repository of Zhongyang-debug, and new methods proposed
    in the paper are implemented by

        - Süleyman Ateş - ates.suleyman@metu.edu.tr
        - Arda Numanoğlu - arda.numanoglu@metu.edu.tr

    The reproduction paper, SFSRNet: Super-resolution for single-channel Audio Source Separation, can be 
    found at: https://www.aaai.org/AAAI22Papers/AAAI-1535.RixenJ.pdf

    The SepFormer model from the paper, Attention is All You Need in Speech Separation, can be found at:
    https://arxiv.org/pdf/2010.13154.pdf

    GitHub repository of Zhongyang-debug including reproduction of SepFormer model, can be found at:
    https://github.com/Zhongyang-debug/Attention-Is-All-You-Need-In-Speech-Separation

"""

from email.mime import audio
import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.module import Module
from torch.autograd import Variable
import math


class Encoder(nn.Module):

    def __init__(self, L, N):

        super(Encoder, self).__init__()

        self.L = L

        self.N = N

        self.Conv1d = nn.Conv1d(in_channels=1,
                                out_channels=N,
                                kernel_size=L,
                                stride=L//2,
                                padding=0,
                                bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.Conv1d(x)

        x = self.ReLU(x)

        return x


class Decoder(nn.Module):

    def __init__(self, L, N):

        super(Decoder, self).__init__()

        self.L = L

        self.N = N

        self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=N,
                                                  out_channels=1,
                                                  kernel_size=L,
                                                  stride=L//2,
                                                  padding=0,
                                                  bias=False)

    def forward(self, x):

        x = self.ConvTranspose1d(x)

        return x


class TransformerEncoderLayer(Module):
    """
        TransformerEncoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application.

        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of intermediate layer, relu or gelu (default=relu).

        Examples:
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            >>> src = torch.rand(10, 32, 512)
            >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0):

        super(TransformerEncoderLayer, self).__init__()

        self.LayerNorm1 = nn.LayerNorm(normalized_shape=d_model)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.LayerNorm2 = nn.LayerNorm(normalized_shape=d_model)

        self.FeedForward = nn.Sequential(nn.Linear(d_model, d_model*2),
                                         nn.ReLU())
        
        # This 2D Convolutional layer added instead of linear layer used in SepFormer.
        # Refer to the p.4 "Separation blocks and multi-loss" part of the paper for details.
        self.Conv = nn.Conv2d(in_channels=d_model*2, out_channels=d_model, kernel_size=3, stride=1, padding=1)

    def forward(self, z1):

        ln_z1 = self.LayerNorm1(z1)

        z2 = self.self_attn(ln_z1, ln_z1, ln_z1, attn_mask=None, key_padding_mask=None)[0]

        lin_out = self.FeedForward(self.LayerNorm2(z2 + z1))

        lin_out = lin_out.permute(2, 1, 0)

        lin_out = lin_out.unsqueeze(0)

        z3 = self.Conv(lin_out)

        z3 = z3.squeeze(0)

        z3 = z3.permute(2, 1, 0)
        z3 = z3 + z2 + z1

        return z3


class Positional_Encoding(nn.Module):
    """
        Implement the positional encoding (PE) function.
        PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
        PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):

        super(Positional_Encoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
            Args:
                input: N x T x D
        """
        length = input.size(1)

        return self.pe[:, :length]


class DPTBlock(nn.Module):

    def __init__(self, input_size, nHead, Local_B):

        super(DPTBlock, self).__init__()

        self.Local_B = Local_B

        self.intra_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=5000)
        self.intra_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.intra_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=0))

        self.inter_PositionalEncoding = Positional_Encoding(d_model=input_size, max_len=5000)
        self.inter_transformer = nn.ModuleList([])
        for i in range(self.Local_B):
            self.inter_transformer.append(TransformerEncoderLayer(d_model=input_size,
                                                                  nhead=nHead,
                                                                  dropout=0))

    def forward(self, z):

        B, N, K, P = z.shape  # torch.Size([1, 64, 250, 130])

        # intra DPT
        row_z = z.permute(0, 3, 2, 1).reshape(B*P, K, N)
        row_z1 = row_z + self.intra_PositionalEncoding(row_z)

        for i in range(self.Local_B):
            row_z1 = self.intra_transformer[i](row_z1.permute(1, 0, 2)).permute(1, 0, 2)

        row_f = row_z1 + row_z
        row_output = row_f.reshape(B, P, K, N).permute(0, 3, 2, 1)

        # inter DPT
        col_z = row_output.permute(0, 2, 3, 1).reshape(B*K, P, N)
        col_z1 = col_z + self.inter_PositionalEncoding(col_z)

        for i in range(self.Local_B):
            col_z1 = self.inter_transformer[i](col_z1.permute(1, 0, 2)).permute(1, 0, 2)

        col_f = col_z1 + col_z
        col_output = col_f.reshape(B, K, P, N).permute(0, 3, 1, 2)

        return col_output


class Separator(nn.Module):

    def __init__(self, N, C, H, K, L, Global_B, Local_B):

        super(Separator, self).__init__()

        self.N = N
        self.C = C
        self.K = K
        self.Global_B = Global_B
        self.Local_B = Local_B
        self.L = L

        self.DPT = nn.ModuleList([])
        for i in range(self.Global_B):
            self.DPT.append(DPTBlock(N, H, self.Local_B))

        self.LayerNorm = nn.LayerNorm(self.N)
        self.Linear1 = nn.Linear(in_features=self.N, out_features=self.N, bias=None)

        self.PReLU = nn.PReLU()
        self.Linear2 = nn.Linear(in_features=self.N, out_features=self.N*2, bias=None)

        self.FeedForward1 = nn.Sequential(nn.Linear(self.N, self.N*2*2),
                                          nn.ReLU(),
                                          nn.Linear(self.N*2*2, self.N))
        self.FeedForward2 = nn.Sequential(nn.Linear(self.N, self.N*2*2),
                                          nn.ReLU(),
                                          nn.Linear(self.N*2*2, self.N))
        self.ReLU = nn.ReLU()

        self.decoder = Decoder(self.L, self.N)

    def forward(self, x_original, rest):
        # Norm + Linear
        x = self.LayerNorm(x_original.permute(0, 2, 1))
        x = self.Linear1(x).permute(0, 2, 1)

        # Chunking
        out, gap = self.split_feature(x, self.K)  # 分块，[B, N, I] -> [B, N, K, S]

        # SepFormer
        # This is the implementation of repeating the separator K times for multi-loss system proposed in the paper.
        # Related explanations can be found at p.4 and in Figure 3 in the paper.
        for i in range(self.Global_B):
            out = self.DPT[i](out)  # [B, N, K, S] -> [B, N, K, S]

            # PReLU + Linear
            out_sepformer = self.PReLU(out)
            out_sepformer = self.Linear2(out_sepformer.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

            B, _, K, S = out_sepformer.shape

            # OverlapAdd
            out_sepformer = out_sepformer.reshape(B, -1, self.C, K, S).permute(0, 2, 1, 3, 4)  # [B, N*C, K, S] -> [B, N, C, K, S]
            out_sepformer = out_sepformer.reshape(B * self.C, -1, K, S)
            out_sepformer = self.merge_feature(out_sepformer, gap)  # [B*C, N, K, S]  -> [B*C, N, I]

            # FFW + ReLU
            out_sepformer = self.FeedForward1(out_sepformer.permute(0, 2, 1))
            out_sepformer = self.FeedForward2(out_sepformer).permute(0, 2, 1)
            masks = self.ReLU(out_sepformer)

            _, N, I = masks.shape

            masks = masks.view(self.C, -1, N, I)  # [C, B, N, I]，torch.Size([2, 1, 64, 16002])

            # Masking
            out_sepformer = [masks[i] * x_original for i in range(self.C)]  # C * ([B, N, I]) * [B, N, I]


            # Decoding
            audio = [self.decoder(out_sepformer[i]) for i in range(self.C)]  # C * [B, 1, T]

            audio[0] = audio[0][:, :, self.L // 2:-(rest + self.L // 2)].contiguous()  # B, 1, T
            audio[1] = audio[1][:, :, self.L // 2:-(rest + self.L // 2)].contiguous()  # B, 1, T
            audio = torch.cat(audio, dim=1)  # [B, C, T]
            yield audio

    def pad_segment(self, input, segment_size):

        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2

        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T


class Sepformer(nn.Module):
    """
        Args:
            C: Number of speakers
            N: Number of filters in autoencoder
            L: Length of the filters in autoencoder
            H: Multi-head
            K: segment size
            R: Number of repeats
    """

    def __init__(self, N=64, C=2, L=4, H=4, K=250, Global_B=2, Local_B=4):

        super(Sepformer, self).__init__()

        self.N = N
        self.C = C
        self.L = L
        self.H = H
        self.K = K
        self.Global_B = Global_B
        self.Local_B = Local_B

        self.encoder = Encoder(self.L, self.N)

        self.separator = Separator(self.N, self.C, self.H, self.K, self.L, self.Global_B, self.Local_B)

        self.decoder = Decoder(self.L, self.N)

    def forward(self, x):

        # Encoding
        x, rest = self.pad_signal(x)  # torch.Size([1, 1, 32006])

        enc_out = self.encoder(x)  # [B, 1, T] -> [B, N, I]，torch.Size([1, 64, 16002])

        # Separetor
        audio = self.separator(enc_out, rest)  # [B, N, I] -> [B*C, N, I]，torch.Size([2, 64, 16002])

        return audio

    def pad_signal(self, input):

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.L - (self.L // 2 + nsample % self.L) % self.L

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.L // 2)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    @classmethod
    def load_model(cls, path):

        package = torch.load(path, map_location=lambda storage, loc: storage)

        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):

        model = cls(N=package['N'], C=package['C'], L=package['L'],
                    H=package['H'], K=package['K'], Global_B=package['Global_B'],
                    Local_B=package['Local_B'])

        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):

        package = {
            # hyper-parameter
            'N': model.N, 'C': model.C, 'L': model.L,
            'H': model.H, 'K': model.K, 'Global_B': model.Global_B,
            'Local_B': model.Local_B,

            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package
