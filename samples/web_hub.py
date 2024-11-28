import re
import uuid
import gradio as gr
import requests
import json
import base64
import os

session = requests.Session()
if not os.path.exists("audios"):
    os.makedirs("audios")

###### 应用列表 ######
def sherpa(ip, filepath):
    url = f"http://{ip}/sherpa/v1/audio/transcriptions"
    files = {'file': open(filepath, 'rb')}
    data = {'response_format': 'text'}
    response = session.post(url, files=files, data=data)
    if response.status_code == 200:
        return response.text
    else:
        return f"请求失败，状态码：{response.status_code}"

######    00    ######
def emotivoice_speech(ip, text, audio_path='', voice='8051', emotion=''):
    url = f"http://{ip}/emotivoice/v1/audio/speech"
    json_data = {
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "emotion": emotion,
        "audio_path": audio_path
    }
    response = session.post(url, json=json_data)
    if response.status_code == 200:
        audio_file = f"audios/{str(uuid.uuid4())}.wav"
        with open(audio_file, 'wb') as f:
            f.write(response.content)
        return audio_file
    else:
        return f"请求失败，状态码：{response.status_code}"

######    01    ######
def emotivoice_translation(ip, src_file, ref_file):
    url = f"http://{ip}/emotivoice/v1/audio/translation"
    files = {
        'file': open(src_file, 'rb'),
        'ref_file': open(ref_file, 'rb')
    }
    response = session.post(url, files=files)
    if response.status_code == 200:
        result = response.json()
        audio_base64 = result.get('text', '')
        audio_data = base64.b64decode(audio_base64)
        audio_file = f"audios/{str(uuid.uuid4())}.wav"
        with open(audio_file, 'wb') as f:
            f.write(audio_data)
        return audio_file
    else:
        return f"请求失败，状态码：{response.status_code}"

######    02    ######
PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]
history = PROMPT
def llm_chat(ip, input_str, image_str, chatbot, model_name):
    global history, PROMPT
    chatbot += [[input_str, ""]]
    history += [{"role": "user", "content": input_str}]
    url = f"http://{ip}/llm_tpu/v1/chat/completions"
    headers = {'Content-Type': 'application/json'}

    if ("minicpmv" in model_name.lower()) and image_str:
        messages = [{"role": "user", "content": [{"type":"text","text":input_str},{"type":"image_url","image_url":{"url":image_str}}]}]
    else:
        messages = history

    json_data = {
        "model": model_name,
        "messages": messages,
        "stream": True
    }
    response = session.post(url, headers=headers, json=json_data, stream=True)
    buffer = ''
    out_str = ''
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line == '':
                continue  # 跳过心跳包或空行
            data_json = json.loads(decoded_line)
            delta = data_json['choices'][0]['delta']
            content = delta.get('content', '')
            if content:
                buffer += content
                match = re.search(r'([.,!?，。！？])', buffer)
                if match:
                    end_index = match.end()
                    to_send = buffer[:end_index]
                    out_str += to_send
                    chatbot[-1][1] += to_send
                    yield chatbot
                    buffer = buffer[end_index:]
    if buffer:
        chatbot[-1][1] += buffer
        out_str += buffer
        yield chatbot
    history += [{"role": "assistant", "content": out_str}]



###### 应用商店 ######
with gr.Blocks() as app_store:
    gr.Markdown("# 欢迎来到AIGC应用商店")
    ip = gr.Textbox(lines=1, label="Host IP", value="localhost:8000")

    with gr.Tab("LLM"):
        gr.Markdown("## 语言模型")
        def reset():
            global history, PROMPT
            history = PROMPT
            return [[None, None]]
        def clear():
            return ""
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(label="Model", choices=["minicpm3", "minicpmv26", "qwen2.5", "phi3"], value="minicpm3")
                chatbot = gr.Chatbot(label="Chat with LLM", height=400)
                with gr.Row():
                    with gr.Column():
                        input_str = gr.Textbox(show_label=False, placeholder="Chat with LLM")
                        with gr.Row():
                            submitBtn = gr.Button("Submit", variant="primary")
                            emptyBtn = gr.Button(value="Clear")
                    with gr.Column():
                        image_str = gr.Textbox(label="image_url", placeholder="Image URL")
        input_str.submit(llm_chat, inputs=[ip, input_str, image_str, chatbot, model_name], outputs=chatbot).then(clear, outputs=input_str)
        submitBtn.click(llm_chat, inputs=[ip, input_str, image_str, chatbot, model_name], outputs=chatbot).then(clear, outputs=input_str)
        emptyBtn.click(reset, outputs=chatbot)

    with gr.Tab("ASR-Sherpa"):
        gr.Markdown("## Sherpa 语音识别")
        audio_file = gr.Audio(type="filepath", label="上传音频文件")
        output_text = gr.Textbox(lines=5, label="识别结果")
        submit_btn = gr.Button("提交")
        submit_btn.click(fn=sherpa, inputs=[ip, audio_file], outputs=output_text)

    with gr.Tab("Emotivoice Speech"):
        gr.Markdown("## Emotivoice 语音合成")
        text = gr.Textbox(lines=5, label="输入文本")
        voice = gr.Textbox(lines=1, label="说话人ID", value="8051")
        emotion = gr.Textbox(lines=1, label="情感提示（可选）")
        ref_audio = gr.Audio(type="filepath", label="参考音频文件")
        audio_output = gr.Audio(label="生成的语音")
        submit_btn = gr.Button("提交")
        def generate_speech(ip, text, ref_audio, voice, emotion):
            result = emotivoice_speech(ip, text, ref_audio, voice, emotion)
            if os.path.isfile(result):
                return result
            else:
                return None
        submit_btn.click(fn=generate_speech, inputs=[ip, text, ref_audio, voice, emotion], outputs=audio_output)

    with gr.Tab("Emotivoice Translation"):
        gr.Markdown("## Emotivoice 音色转换")
        src_audio = gr.Audio(type="filepath", label="源音频文件")
        ref_audio = gr.Audio(type="filepath", label="参考音频文件")
        audio_output = gr.Audio(label="转换后的音频")
        submit_btn = gr.Button("提交")
        def translate_voice(ip, src_audio, ref_audio):
            result = emotivoice_translation(ip, src_audio, ref_audio)
            if os.path.isfile(result):
                return result
            else:
                return None
        submit_btn.click(fn=translate_voice, inputs=[ip, src_audio, ref_audio], outputs=audio_output)

app_store.launch(server_name="0.0.0.0", server_port=5000, inbrowser=True)