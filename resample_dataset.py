import argparse
import os

from pydub import AudioSegment


def resample_audio(file_path, output_file, c=1, hz=48000,format='wav'):
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

def cut_audio(file_path, start_time, end_time, output_file,format='wav'):
    try:
        # 读取音频文件  
        audio = AudioSegment.from_file(file_path, format=format)  
        # 指定裁剪的起始和结束时间（以毫秒为单位）  
        if len(audio) > (end_time - start_time):
            cropped_audio = audio[start_time:end_time]  
            # 将裁剪后的音频保存为新的文件  
            cropped_audio.export(output_file, format='wav')
            #return output_file
    except Exception as e:
        print(f"Error: {e}")
        #return None



def change_amplitude(file_path,output_file,format="wav"):
  
    # 加载音频文件  
    audio = AudioSegment.from_file(file_path, format=format)  
  
    # 改变幅度  
    # gain_amount为幅度值，可以为负数来降低音频的幅度  
    audio = audio.apply_gain(gain_amount=10)  
    
    # 保存处理后的音频文件  
    audio.export(output_file, format=format)
    
if __name__ == '__main__':
    resample_audio("/sda1/lanxin/MP-SENet-main/test.mp3","/sda1/lanxin/MP-SENet-main/test.wav",hz=16000)
    # parser = argparse.ArgumentParser(description="Resample audio files")
    # parser.add_argument('--data_path', default="/sda1/lanxin/MP-SENet-main/VoiceBank_DEMAND/wavs_noisy_/", type=str)
    # parser.add_argument('--out_path', default="/sda1/lanxin/MP-SENet-main/VoiceBank_DEMAND/wavs_noisy/", type=str)
    # parser.add_argument('--start_time', default=0, type=int)
    # parser.add_argument('--end_time', default=10000, type=int)
    # args = parser.parse_args()
    # data_root = args.data_path
    # output_dir = args.out_path
    # end_time = args.end_time
    # start_time = args.start_time
    # data_dirs = os.listdir(data_root)
    # for data_dir in data_dirs:
    #     # files = os.listdir(os.path.join(data_root, data_dir))
    #     # if not os.path.exists(os.path.join(output_dir, data_dir)):
    #     #     os.makedirs(os.path.join(output_dir, data_dir))
    #     # files = os.listdir(os.path.join(data_root, data_dir))
    #     if not os.path.exists(os.path.join(output_dir)):
    #         os.makedirs(os.path.join(output_dir))
    #     #for audio_file in data_dir:
    #         # if resample_audio(os.path.join(data_root, data_dir, audio_file),
    #         #                os.path.join(output_dir, data_dir, audio_file)):#重采样成功之后进行cut
    #     audio_file = data_dir
    #     if audio_file.endswith('.wav'):
    #         cut_audio(os.path.join(data_root, audio_file), 
    #                 start_time, end_time,os.path.join(output_dir, audio_file)) 

    # # 裁剪音频
    # data_dirs = "/sda1/lanxin/MP-SENet-main/VoiceBank_DEMAND/wavs_noisy_lx"
    # files = os.listdir(data_dirs)
    # for file in files:    
    #     # 读取原始音频文件  
    #     audio = AudioSegment.from_file(os.path.join(data_dirs,file), format="wav")  
    #     name = file.split('.')[0]
    #     # 裁剪成多个10秒的短音频文件  
    #     for i in range(0, len(audio), 10000): # 每10秒一个片段  
    #         if i + 10000 < len(audio):  
    #             segment = audio[i:i+10000]  
    #         else: # 处理最后一小段不足10秒的音频文件  
    #             segment = audio[i:]  
            
    #         segment.export(os.path.join("/sda1/lanxin/MP-SENet-main/VoiceBank_DEMAND/wavs_noisy_lx/", name +f"_segment{i//10000}.wav"), format="wav") # 导出每个片段为单独的音频文件
   
   
    ## 合并音频

  
    # # 读取多个10秒的短音频文件  
    # data_dirs = "/sda1/lanxin/MP-SENet-main/generated_files"
    # save_dir = '/sda1/lanxin/MP-SENet-main/VoiceBank_DEMAND/TEST/out'
    # files = sorted(os.listdir(data_dirs))
    # files_gp = {}
    # for file in files:
    #     G,N,S = file.split('_')[0],file.split('_')[2],file[-5]
    #     first_letter = G+'_'+N 
    #     # 如果字母在letters字典中不存在，则添加进去  
    #     if first_letter not in files_gp:  
    #         files_gp[first_letter] = []  
    #     # 将文件名添加到letters字典中  
    #     files_gp[first_letter].append(file) 
    # #print(files_gp) 
    
    # for k, v in files_gp.items(): 
    #     segments = []  
    #     for filename in v:
    #         segment = AudioSegment.from_file(os.path.join(data_dirs,filename), format="wav")  
    #         segments.append(segment)  

    #     # 将多个短音频文件合并成一个音频文件  
    #     merged_audio = segments[0]  
    #     for i in range(1, len(segments)):  
    #         merged_audio = merged_audio + segments[i]  
        
    #     # 导出合并后的音频文件  
    #     merged_audio.export(os.path.join(save_dir,k+'.wav'), format="wav")