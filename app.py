#基于gradio的web框架 实现 演示demo

import gradio as gr
import os

from interface import run
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

HOT_MODELS = [
    "\N{fire}MPSENET"
    ]

Instrcution = {
    "MPSENET":"MPSENET支持采样率为16khz的音频文件，速度较快，但是处理长时间的音频处理很容易显存不够，长音频裁剪后处理再拼接，整体效果可能下降。"
    }

    
def greet(noisy,segment,session):
    
    save_path = os.path.join(session['root_path'],session['out_path'],session['model'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    name = os.path.basename(noisy)
    enhance = os.path.join(save_path, name.split('.')[0]+'.wav')
    infos = Instrcution[session['model']]

    if session['model'] =="MPSENET":
        try:
            run(noisy, save_path,segment)
            #print(noisy, save_path)
        except Exception as e:
            infos = "发生错误:" + str(e)
            if not os.path.exists(enhance):
                enhance = noisy
   
    
    return enhance, infos

def choice_user_voice(choice, session):
    session['model'] = choice[1:]
    return session


with gr.Blocks(    
    title="Speech Enhancement",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),) as demo:

    gr.Markdown("# Speech Enhancement")
    
    session = gr.State({
    'root_path': 'examples', #修改这里即可
    'out_path': 'enhance',
    'model': 'MPSENET'
    })
    
    
    with gr.Row():

        with gr.Column():
            noisy = gr.Audio(
                type='filepath', label='Target Audio',format='wav')

        with gr.Column():
            output = gr.Audio(label="Output Audio",type="filepath",format='wav')
            
        
    with gr.Row():
        with gr.Column():
            gr.Examples(
                label="Audio Examples For All Models",
                examples=[
                    os.path.join(os.path.dirname(__file__),
                                    "examples/speech_with_noise.wav"),
                    os.path.join(os.path.dirname(__file__),
                                    "examples/speech_with_noise1.wav"),
                ],
                inputs=[noisy],
                fn=greet,
            )
        with gr.Column():
            user_models = gr.Radio(label="models", choices=HOT_MODELS, type="value", value=HOT_MODELS[0])
            segment = gr.Slider(minimum=16000, maximum=320000, step=1000, value=160000, label="音频裁剪长度")
            info = gr.Textbox(label="Notes", type='text',value="MPSENET可支持在线上传Audio。\n ")
    with gr.Row():
        btn = gr.Button(value="Enhancement",variant="primary")
    with gr.Row():
        output_txt = gr.Textbox(label="Instructions")
    user_models.change(fn=choice_user_voice, inputs=[user_models, session], outputs=session)

    
    btn.click(fn=greet, inputs=[noisy,segment,session],outputs=[output,output_txt])
    

demo.launch(share=True)
