# from modelscope.utils.constant import ModelFile,Tasks
# from modelscope.metainfo import Models
# from modelscope.models.base import TorchModel
# from modelscope.models.builder import MODELS
# from modelscope.models.base import Tensor

import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F
from spafe.fbanks import linear_fbanks
from typing import List,Dict
import os

def max_neg_value(t):
    return -torch.finfo(t.dtype).max


class ASA(nn.Module):
    def __init__(self, c=64, causal=True):
        super(ASA, self).__init__()
        self.d_c = c//4
        self.f_qkv = nn.Sequential(
            nn.Conv2d(c, self.d_c*3, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.d_c*3),
            nn.PReLU(self.d_c*3),
        )
        self.t_qk = nn.Sequential(
            nn.Conv2d(c, self.d_c*2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.d_c*2),
            nn.PReLU(self.d_c*2),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(self.d_c, c, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.causal = causal

    def forward(self, inp):
        """
        inp: B C F T
        """
        # f-attention
        f_qkv = self.f_qkv(inp)
        qf, kf, v = tuple(einops.rearrange(
            f_qkv, "b (c k) f t->k b c f t", k=3))
        f_score = torch.einsum("bcft,bcyt->btfy", qf, kf) / (self.d_c**0.5)
        f_score = f_score.softmax(dim=-1)
        f_out = torch.einsum('btfy,bcyt->bcft', [f_score, v])
        # t-attention
        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
        t_score = torch.einsum('bcft,bcfy->bfty', [qt, kt]) / (self.d_c**0.5)
        mask_value = max_neg_value(t_score)
        if self.causal:
            i, j = t_score.shape[-2:]
            mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
            t_score.masked_fill_(mask, mask_value)
        t_score = t_score.softmax(dim=-1)
        t_out = torch.einsum('bfty,bcfy->bcft', [t_score, f_out])
        out = self.proj(t_out)
        return out + inp


class FD(nn.Module):
    def __init__(self, cin, cout, K=(7, 1), S=(4, 1), P=(2, 0)):
        super(FD, self).__init__()
        self.fd = nn.Sequential(
            nn.Conv2d(cin, cout, K, S, P, groups=2),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout)
        )

    def forward(self, x):
        return self.fd(x)


class FU(nn.Module):
    def __init__(self, cin, cout, K=(7, 1), S=(4, 1), P=(2, 0), O=(1, 0)):
        super(FU, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin*2, cin, (1, 1)),
            nn.BatchNorm2d(cin),
            nn.Tanh(),
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 1)),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout),
        )
        #  22/06/13 update, add groups = 2
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(cout, cout, K, S, P, O, groups=2),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout)
        )

    def forward(self, fu, fd):
        """
        fu, fd: B C F T
        """
        outs = self.pconv1(torch.cat([fu, fd], dim=1))*fd
        outs = self.pconv2(outs)
        outs = self.conv3(outs)
        return outs
    



class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        return out


def complex_cat(inps, dim=1):
    reals, imags = [], []
    for inp in inps:
        real, imag = inp.chunk(2, dim)
        reals.append(real)
        imags.append(imag)
    reals = torch.cat(reals, dim)
    imags = torch.cat(imags, dim)
    return reals, imags


class ComplexLinearProjection(nn.Module):
    def __init__(self, cin):
        super(ComplexLinearProjection, self).__init__()
        self.clp = ComplexConv2d(cin, cin)

    def forward(self, real, imag):
        """
        real, imag: B C F T
        """
        inputs = torch.cat([real, imag], 1)
        outputs = self.clp(inputs)
        real, imag = outputs.chunk(2, dim=1)
        outputs = torch.sqrt(real**2+imag**2+1e-8)
        return outputs


class PhaseEncoder(nn.Module):
    def __init__(self, cout, n_sig, cin=2, alpha=0.5):
        super(PhaseEncoder, self).__init__()
        self.complexnn = nn.ModuleList()
        for _ in range(n_sig):
            self.complexnn.append(
                nn.Sequential(
                    nn.ConstantPad2d((2, 0, 0, 0), 0.0),
                    ComplexConv2d(cin, cout, (1, 3))
                )
            )
        self.clp = ComplexLinearProjection(cout*n_sig)
        self.alpha = alpha

    def forward(self, cspecs):
        """
        cspec: B C F T
        """
        outs = []
        for idx, layer in enumerate(self.complexnn):
            outs.append(layer(cspecs[idx]))
        real, imag = complex_cat(outs, dim=1)
        amp = self.clp(real, imag)
        return amp**self.alpha



def mag_pha_stft(y, n_fft=400, hop_size=100, win_size=400, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

class STFT(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, win_type):
        super(STFT, self).__init__()
        self.win, self.hop = win_len, hop_len
        self.nfft = fft_len
        window = {
            "hann": torch.hann_window(win_len),
            "hamm": torch.hamming_window(win_len),
        }
        assert win_type in window.keys()
        self.window = window[win_type]

    def transform(self, inp):
        """
        inp: B N
        """
        cspec = torch.stft(inp, self.nfft, self.hop, self.win,
                        self.window.to(inp.device), return_complex=False)
        cspec = einops.rearrange(cspec, "b f t c -> b c f t")
        return cspec

    def inverse(self, real, imag):
        """
        real, imag: B F T
        """
        inp = torch.stack([real, imag], dim=-1)
        return torch.istft(inp, self.nfft, self.hop, self.win, self.window.to(real.device))



class TFCM_Block(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 dila=1,
                 causal=True,
                 ):
        super(TFCM_Block, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=(1, 1)),
            nn.BatchNorm2d(cin),
            nn.PReLU(cin),
        )
        dila_pad = dila * (K[1] - 1)
        if causal:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad, 0, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                nn.BatchNorm2d(cin),
                nn.PReLU(cin)
            )
        else:
            # update 22/06/21, add groups for non-casual
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad//2, dila_pad//2, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                nn.BatchNorm2d(cin),
                nn.PReLU(cin)
            )
        self.pconv2 = nn.Conv2d(cin, cin, kernel_size=(1, 1))
        self.causal = causal
        self.dila_pad = dila_pad

    def forward(self, inps):
        """
            inp: B x C x F x T
        """
        outs = self.pconv1(inps)
        outs = self.dila_conv(outs)
        outs = self.pconv2(outs)
        return outs + inps


class TFCM(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 tfcm_layer=6,
                 causal=True,
                 ):
        super(TFCM, self).__init__()
        self.tfcm = nn.ModuleList()
        for idx in range(tfcm_layer):
            self.tfcm.append(
                TFCM_Block(cin, K, 2**idx, causal=causal)
            )

    def forward(self, inp):
        out = inp
        for idx in range(len(self.tfcm)):
            out = self.tfcm[idx](out)
        return out


class Banks(nn.Module):
    def __init__(self, nfilters, nfft, fs, low_freq=0.0, high_freq=None, learnable=False):
        super(Banks, self).__init__()
        self.nfilters, self.nfft, self.fs = nfilters, nfft, fs
        filter, _ = linear_fbanks.linear_filter_banks(
            nfilts=self.nfilters,
            nfft=self.nfft,
            low_freq=low_freq,
            high_freq=high_freq,
            fs=self.fs,
        )
        filter = torch.from_numpy(filter).float()
        if not learnable:
            #  30% energy compensation.
            self.register_buffer('filter', filter*1.3)
            self.register_buffer('filter_inv', torch.pinverse(filter))
        else:
            self.filter = nn.Parameter(filter)
            self.filter_inv = nn.Parameter(torch.pinverse(filter))

    def amp2bank(self, amp):
        amp_feature = torch.einsum("bcft,kf->bckt", amp, self.filter)
        return amp_feature

    def bank2amp(self, inputs):
        return torch.einsum("bckt,fk->bcft", inputs, self.filter_inv)

def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]
 
# # MODELS.register_module部分是为了将模型注册进Registry中，这样可以在Model.from_pretrained中被使用到。在模型被初始化时，模型的构造方法就会被调用。
# @MODELS.register_module(    
#     Tasks.acoustic_noise_suppression,
#     module_name=Models.speech_mtfaa_ans_cirm_16k)
# class MTFAADecorator(TorchModel):
#     r""" A decorator of MTFAA for integrating into modelscope framework """
#     # model_dir作为第一个参数

#     def __init__(self, model_dir: str, *args, **kwargs):
#         """initialize the frcrn model from the `model_dir` path.

#         Args:
#             model_dir (str): the model path.
#         """
#         super().__init__(model_dir, *args, **kwargs)
#         self.model = MTFAANet(*args, **kwargs)
#         model_bin_file = os.path.join(model_dir,
#                                       ModelFile.TORCH_MODEL_BIN_FILE)
#         if os.path.exists(model_bin_file):
#             checkpoint = torch.load(
#                 model_bin_file, map_location=torch.device('cpu'))
#             if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#                 # the new trained model by user is based on FRCRNDecorator
#                 self.load_state_dict(checkpoint['state_dict'])
#             else:
#                 # The released model on Modelscope is based on FRCRN
#                 self.model.load_state_dict(checkpoint, strict=False)

#     def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         result_list = self.model.forward(inputs['noisy'])
#         output = {
#             'spec_l1': result_list[0],
#             'wav_l1': result_list[1],
#             'mask_l1': result_list[2],
#             'spec_l2': result_list[3],
#             'wav_l2': result_list[4],
#             'mask_l2': result_list[5]
#         }
#         if 'clean' in inputs:
#             mix_result = self.model.loss(
#                 inputs['noisy'], inputs['clean'], result_list, mode='Mix')
#             output.update(mix_result)
#             sisnr_result = self.model.loss(
#                 inputs['noisy'], inputs['clean'], result_list, mode='SiSNR')
#             output.update(sisnr_result)
#             # logger hooker will use items under 'log_vars'
#             output['log_vars'] = {k: mix_result[k].item() for k in mix_result}
#             output['log_vars'].update(
#                 {k: sisnr_result[k].item()
#                  for k in sisnr_result})
#         return output
    
class MTFAANet(nn.Module):
    
    def __init__(self, 
                 n_sig=1,
                 PEc=4,
                 Co="48,96,192",
                 O="1,1,1",
                 causal=True,
                 bottleneck_layer=2,
                 tfcm_layer=6,
                 mag_f_dim=3,
                 win_len=400,
                 win_hop=100,
                 nerb=256,
                 sr=16000,
                 win_type="hann",):
        super().__init__()
        self.PE = PhaseEncoder(PEc, n_sig)
        # 32ms @ 48kHz
        self.stft = STFT(win_len, win_hop, win_len, win_type)
        self.ERB = Banks(nerb, win_len, sr)
        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        C_en = [PEc//2*n_sig] + parse_1dstr(Co)
        C_de = [4] + parse_1dstr(Co)
        O = parse_1dstr(O)
        for idx in range(len(C_en)-1):
            self.encoder_fd.append(
                FD(C_en[idx], C_en[idx+1]),
            )
            self.encoder_bn.append(
                nn.Sequential(
                    TFCM(C_en[idx+1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[idx+1], causal=causal),
                )
            )

        for idx in range(bottleneck_layer):
            self.bottleneck.append(
                nn.Sequential(
                    TFCM(C_en[-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[-1], causal=causal),
                )
            )

        for idx in range(len(C_de)-1, 0, -1):
            self.decoder_fu.append(
                FU(C_de[idx], C_de[idx-1], O=(O[idx-1], 0)),
            )
            self.decoder_bn.append(
                nn.Sequential(
                    TFCM(C_de[idx-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_de[idx-1], causal=causal),
                )
            )
        # MEA is causal, so mag_t_dim = 1.
        self.mag_mask = nn.Conv2d(
            4, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = torch.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim
        
    def forward(self, sigs):
        """
        sigs: list [B N] of len(sigs)
        """
        cspecs = []
        for sig in sigs:
            cspecs.append(self.stft.transform(sig))
        # D / E ?
        D_cspec = cspecs[0]
        mag = torch.norm(D_cspec, dim=1)
        pha = torch.atan2(D_cspec[:, -1, ...], D_cspec[:, 0, ...])
        out = self.ERB.amp2bank(self.PE(cspecs))
        encoder_out = []
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            encoder_out.append(out)
            out = self.encoder_bn[idx](out)

        for idx in range(len(self.bottleneck)):
            out = self.bottleneck[idx](out)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1-idx])
            out = self.decoder_bn[idx](out)
        out = self.ERB.bank2amp(out)
        # stage 1
        mag_mask = self.mag_mask(out)
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.sigmoid()
        mag = mag.sum(dim=1)
        # stage 2
        real_mask = self.real_mask(out).squeeze(1)
        imag_mask = self.imag_mask(out).squeeze(1)

        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, 1e-10))
        pha_mask = torch.atan2(imag_mask+1e-10, real_mask+1e-10)
        real = mag * mag_mask.tanh() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.tanh() * torch.sin(pha+pha_mask)
        eps = 1e-7
        amplitude = torch.sqrt(real**2+imag**2 + eps)
        phase = torch.atan2(imag + eps,real + eps)
        return mag, amplitude, phase, self.stft.inverse(real, imag)#torch.stack([real, imag], dim=1), self.stft.inverse(real, imag)

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

def test_nnet():
    # noise supression (microphone, )
    nnet = MTFAANet(n_sig=1)
    inp = torch.randn(1, 16000)
    mag, cspec, wav = nnet([inp])
    print(mag.shape, cspec.shape, wav.shape)
    # echo cancellation (microphone, error, reference,)
    nnet = MTFAANet(n_sig=3)
    mag, cspec, wav = nnet([inp, inp, inp])
    print(mag.shape, cspec.shape, wav.shape)


def test_mac():
    from thop import profile, clever_format
    import torch as th
    nnet = MTFAANet(n_sig=3)
    # hop=8ms, win=32ms@48KHz, process 1s.
    inp = th.randn(1, 48000)
    # inp = th.randn(1, 2, 769, 126)
    macs, params = profile(nnet, inputs=([inp, inp, inp],), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('macs: ', macs)
    print('params: ', params)


if __name__ == "__main__":
    test_nnet()
    #test_mac()
    