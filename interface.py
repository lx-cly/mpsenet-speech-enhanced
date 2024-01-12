#mp-senet模型 对接 接口


#导入包
import glob
import os
import argparse
import json
import torch
import librosa
import shutil
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
from pydub import AudioSegment


#音频预处理 -- 单通道、16Khz采样
def resample_audio(file_path, output_file, c=1, hz=16000,format='wav'):
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(c)  # 设置声道数为双声道
        audio = audio.set_frame_rate(hz)  # 设置采样频率为 44100Hz
        # 新文件名，保存为 WAV 格式
        audio.export(output_file, format=format)
        return output_file
    except Exception as e:
        print(f"Error: {e}")
        return None
    


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def inference(input_noisy, output_dir, checkpoint_file, json_config, segment_size):
    # 读取模型配置
    h = AttrDict(json_config)
    # 设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
        
    # 模型结构
    model = MPNet(h).to(device)

    # 加载权重文件
    state_dict = load_checkpoint(checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(output_dir, exist_ok=True)

    #开始推理
    model.eval()
    
    #获取输入音频路径
    if os.path.isdir(input_noisy): #音频文件夹
        test_indexes = glob.glob(f"{input_noisy}/**")  
    else:
        test_indexes = [input_noisy] #单个音频文件
        
    with torch.no_grad():
        for _ , index in enumerate(test_indexes):
            
            resample_audio(index,index,hz=16000) #覆盖

            noisy_wav, _ = librosa.load(index, sr=h.sampling_rate)# , h.sampling_rate
            audio_start = 0

            noisy_wav = torch.FloatTensor(noisy_wav)
            enhance_wav = torch.zeros_like(noisy_wav)
            while (noisy_wav.size(0) - audio_start) > 0: #截断分开处理
                noisy_audio = noisy_wav[audio_start: audio_start + segment_size]
                
                noisy_audio = torch.FloatTensor(noisy_audio).to(device)
                norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0)).to(device)
                noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)
                noisy_amp, noisy_pha, _ = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
                audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                audio_g = audio_g / norm_factor
                if audio_g.size(1) < segment_size:
                    enhance_wav[audio_start:audio_start+audio_g.size(1)] = audio_g
                else:
                    enhance_wav[audio_start: audio_start+segment_size] = audio_g
                audio_start += segment_size

            output_file = os.path.join(output_dir, os.path.basename(index))

            sf.write(output_file, enhance_wav.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


# use command: python interface.py --input_noisy {path} --output_dir {path} 

def run(input_noisy ='examples/speech_with_noise.wav', output_dir='generated_files',segment_size = 200000):
    print('Initializing Inference Process..')
    checkpoint_file = "weights/mpsenet" #fix model path
    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    
    inference(input_noisy, output_dir, checkpoint_file, json_config, segment_size)
    
    torch.cuda.empty_cache() #清空显存
    

if __name__ == '__main__':
# use command: python interface.py --input_noisy {path} --output_dir {path} 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy', default='examples/speech_with_noise.wav')
    parser.add_argument('--output_dir', default='generated_files')
    args = parser.parse_args()
    run(args.input_noisy, args.output_dir)