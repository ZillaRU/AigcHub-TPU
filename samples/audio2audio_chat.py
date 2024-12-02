
import json
import os
import uuid
import gradio as gr
import requests
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence
import queue
import threading
import re
import platform


system = platform.system()
session = requests.Session()


if system == 'Windows':
    from playsound import playsound
elif system == 'Linux':
    import subprocess
else:
    raise OSError(f"Unsupported operating system: {system}")

if not os.path.exists("audios"):
    os.makedirs("audios")


def play_sound(audio_file):
    global system
    if system == 'Windows':
        playsound(audio_file)
    else:
        subprocess.run(['aplay', audio_file])

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    processed_audio = AudioSegment.empty()

    for chunk in chunks:
        processed_audio += chunk

    processed_audio.export("./audios/processed_audio.wav", format="wav")
    return "./audios/processed_audio.wav"

# 初始化历史记录
PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]
history = PROMPT 
conversation_count = 0

# 语音播放队列
audio_queue = queue.Queue()

def audio_player():
    while True:
        audio_file = audio_queue.get()
        if (audio_file is None):
            break
        play_sound(audio_file)
        audio_queue.task_done()

# 启动语音播放线程
threading.Thread(target=audio_player, daemon=True).start()

def a2t(ip, file_path):
    url = f"http://{ip}:8000/sherpa/v1/audio/transcriptions"
    files = {'file': open(file_path, 'rb')}
    data = {'response_format': 'json'}
    response = session.post(url, files=files, data=data)
    return response.json()['text']

def llm(ip, messages):
    url = f"http://{ip}:8000/llm_tpu/v1/chat/completions"
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "minicpm3",
        "messages": messages,
        "stream": True
    }
    st = time.time()
    response = session.post(url, headers=headers, json=data, stream=True)
    buffer = ''
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line == '':
                continue  # 跳过心跳包或空行
            # 解析JSON数据
            data_json = json.loads(decoded_line)
            # 提取内容
            delta = data_json['choices'][0]['delta']
            content = delta.get('content', '')
            if content:
                buffer += content
                # 检查是否有句末标点符号
                match = re.search(r'([.,!?，。！？])', buffer)
                if match:
                    end_index = match.end()
                    to_send = buffer[:end_index]
                    threading.Thread(target=t2a, args=(ip, to_send,)).start()
                    yield to_send
                    print(f"LLM Response: {to_send}, Time: {time.time() - st:.2f}s")
                    buffer = buffer[end_index:]

    # 处理剩余的缓冲区内容
    if buffer:
        threading.Thread(target=t2a, args=(buffer,)).start()
        yield buffer
        print(f"LLM Response: {buffer}, Time: {time.time() - st:.2f}s")

def t2a(ip, txt):
    url = f"http://{ip}:8000/emotivoice/v1/audio/speech"
    headers = {'Content-Type': 'application/json'}
    data = {"input": txt}
    response = session.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        audio_content = response.content
        audio_format = response.headers.get('Content-Type').split('/')[-1]
        audio_file_path = f"./audios/{str(uuid.uuid4())}.{audio_format}"
        
        with open(audio_file_path, 'wb') as audio_file:
            audio_file.write(audio_content)
        
        audio_queue.put(audio_file_path)
    else:
        print(f"Error: {response.status_code}, {response.text}")

def process_audio(ip, file_path):
    st = time.time()
    global history, conversation_count

    # Step 0: Preprocess Audio
    file_path = preprocess_audio(file_path)
    # Step 1: Audio to Text
    text = a2t(ip, file_path)
    print(f"Transcribed Text: {text}， Time: {time.time() - st}")
    
    # Step 2: Update history with user input
    history.append({"role": "user", "content": text})

    # Step 3: Text to LLM Response (Streamed)
    response_text = ""
    for partial_response in llm(ip, history):
        response_text += partial_response

    # Step 4: Update history with assistant response
    history.append({"role": "assistant", "content": response_text})

    # Step 5: Increment conversation count and reset history if needed
    conversation_count += 1
    if conversation_count >= 5:
        history = PROMPT
        conversation_count = 0

    return response_text

iface = gr.Interface(fn=process_audio, inputs=[gr.Textbox(lines=1, label="Host IP", value="localhost"), gr.Audio(type="filepath")], outputs="text")
iface.launch(server_name="0.0.0.0", server_port=5000, inbrowser=True)