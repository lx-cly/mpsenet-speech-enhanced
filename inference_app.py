from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MPNet
import soundfile as sf

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    # with open(a.input_test_file, 'r', encoding='utf-8') as fi:
    #     test_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()
    test_indexes = [a.input_noisy_wavs_file]#os.listdir(a.input_noisy_wavs_dir)
    with torch.no_grad():
        for _ , index in enumerate(test_indexes):
            # print(index)
            # noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index),sr=h.sampling_rate)# , h.sampling_rate
            # from resample_dataset import resample_audio
            # resample_audio(noisy,tmp_nosiy,hz=16000)
            # librosa.load(tmp_nosiy,sr=h.sampling_rate)
            noisy_wav, _ = librosa.load(index, sr=h.sampling_rate)# , h.sampling_rate
            audio_start = 0

            noisy_wav = torch.FloatTensor(noisy_wav)
            enhance_wav = torch.zeros_like(noisy_wav)
            while (noisy_wav.size(0) - audio_start) > 0:
                noisy_audio = noisy_wav[audio_start: audio_start+a.segment_size]
                
                noisy_audio = torch.FloatTensor(noisy_audio).to(device)
                norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0)).to(device)
                noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)
                noisy_amp, noisy_pha, _ = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
                audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                audio_g = audio_g / norm_factor
                if audio_g.size(1) < a.segment_size:
                    enhance_wav[audio_start:audio_start+audio_g.size(1)] = audio_g
                else:
                    enhance_wav[audio_start: audio_start+a.segment_size] = audio_g
                audio_start += a.segment_size

            output_file = os.path.join(a.output_dir, os.path.basename(index))

            sf.write(output_file, enhance_wav.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_file', default='/sda1/lanxin/MP-SENet-main/test.wav')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', default="/sda1/lanxin/MP-SENet-main/best_ckpt/g_02050000_lx") #g_02090000
    parser.add_argument('--segment_size', default=200000, type=int) #长度 
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

