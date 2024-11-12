import gradio as gr
import subprocess
import re
import json
import os

STANDALONE = False

sherpa_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/repo/sherpa"

cmd = f"{sherpa_dir}/build/bin/sherpa-onnx --tokens={sherpa_dir}/models/tokens.txt --zipformer2-ctc-model={sherpa_dir}/models/zipformer2_ctc_F32.bmodel "

def run_shell_command(command):
    pattern = re.compile(r'\{.*?\}')
    dictionaries = []
    
    with subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            matches = pattern.findall(line)
            for match in matches:
                try:
                    # 将匹配的字符串转换为字典
                    dict_obj = json.loads(match)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {match}")
    return dict_obj["text"]

def sherpa_main(audio_path):
    print(audio_path)
    print(cmd + audio_path)
    tt = run_shell_command(cmd + audio_path)
    print(tt)
    return tt

app = gr.Interface(fn=sherpa_main, inputs=gr.Audio(type="filepath"), outputs=gr.Textbox(lines=7, interactive=True), title="Sherpa ASR", description="This is a speech recognition model that converts audio to text using the Sherpa model.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app.queue().launch(server_port=args.port, inbrowser=True)

