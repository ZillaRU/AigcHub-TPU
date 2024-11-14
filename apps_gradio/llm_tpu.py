import gradio as gr
import os
import sys
import psutil
import signal
import platform
from subprocess import Popen
import socket

STANDALONE = False
system = platform.system()
Processing = []
port_to_use = 5001

llm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/repo/llm_tpu"


def kill_process(pid):
    if system == "Windows":
        cmd = f"taskkill /t /f /pid {pid}"
        os.system(cmd)
    else:
        kill_proc_tree(pid)

def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)
        except OSError:
            pass

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0
    
def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

host_ip = get_host_ip()

def run_shell_command(command, host_ip, port_to_use):
    process = Popen(command, cwd=llm_dir)
    Processing.append(process)
    mss = f"============================================================= \n Subprocess running on:  http://{host_ip}:{port_to_use} \n Please wait for the models to load, which may take minutes. \n============================================================="
    print("\n" + mss + "\n")
    return mss


def shutdown_process():
    for process in Processing:
        kill_process(process.pid)
        Processing.remove(process)
        mss = "========================================= \n Subprocess has been shut down. \n========================================="
        print("\n" + mss + "\n")
        yield mss

def launch(model_path, temperature, top_p, repeat_penalty, repeat_last_n, max_new_tokens, generation_mode, prompt_mode):
    global Processing, port_to_use

    while not is_port_available(port_to_use):
        port_to_use += 1

    model_type, tokenizer_path = get_model_type(model_path)
    model_path = f"bmodels/{model_path}"
    tokenizer_path = f"models/{tokenizer_path}"

    cmd = [
        sys.executable, f"models/{model_type}/python_demo/web_demo.py",
        '--model_path', model_path,
        '--tokenizer_path', tokenizer_path,
        '--temperature', str(temperature),
        '--top_p', str(top_p),
        '--repeat_penalty', str(repeat_penalty),
        '--repeat_last_n', str(repeat_last_n),
        '--max_new_tokens', str(max_new_tokens),
        '--generation_mode', generation_mode,
        '--prompt_mode', prompt_mode,
        '--port', str(port_to_use)
    ]

    return run_shell_command(cmd, host_ip, port_to_use)


def get_model_type(model_path):
    if "phi3" in model_path.lower():
        tokenizer_path = "Phi3/support/token_config"
        return "Phi3", tokenizer_path
    elif "qwen1.5" in model_path.lower():
        tokenizer_path = "Qwen1_5/token_config"
        return "Qwen1_5", tokenizer_path
    elif "qwen2.5" in model_path.lower():
        tokenizer_path = "Qwen2_5/support/token_config"
        return "Qwen2_5", tokenizer_path
    else:
        raise ValueError("Unsupported model")
    

with gr.Blocks() as demo:
    gr.Markdown("# ChatBox")
    model_choices = os.listdir(f"{llm_dir}/bmodels") if os.path.exists(f"{llm_dir}/bmodels") else ["Not initialized"]
    model_path = gr.Dropdown(
        label="Model",
        choices=model_choices,
        value=model_choices[0] if model_choices else None
    )
    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=1.0)
    top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=1.0)
    repeat_penalty = gr.Slider(label="Repeat Penalty", minimum=0.0, maximum=2.0, value=1.0)
    repeat_last_n = gr.Number(label="Repeat Last N", value=32)
    max_new_tokens = gr.Number(label="Max New Tokens", value=1024)
    generation_mode = gr.Radio(
        label="Generation Mode",
        choices=["greedy", "penalty_sample"],
        value="greedy"
    )
    prompt_mode = gr.Radio(
        label="Prompt Mode",
        choices=["prompted", "unprompted"],
        value="prompted"
    )


    launch_button = gr.Button("Launch")
    shutdown_button = gr.Button("Shutdown", visible=False)
    output_text = gr.Textbox(label="Shell Output", lines=5)  # 文本框用于显示shell输出

    def launch_and_toggle():
        return gr.update(visible=False), gr.update(visible=True)

    def shutdown_and_toggle():
        return gr.update(visible=True), gr.update(visible=False)

    launch_button.click(
        fn=launch,
        inputs=[model_path, temperature, top_p, repeat_penalty, repeat_last_n, max_new_tokens, generation_mode, prompt_mode],
        outputs=[output_text]
    ).then(launch_and_toggle, inputs=None, outputs=[launch_button, shutdown_button])

    shutdown_button.click(
        fn=shutdown_process,
        inputs=None,
        outputs=[output_text]
    ).then(shutdown_and_toggle, inputs=None, outputs=[launch_button, shutdown_button])

# while not is_port_available(port_to_use):
#     port_to_use += 1
# print(f"Running on: http://{host_ip}:{port_to_use}")
# demo.launch(server_name="0.0.0.0", inbrowser=True, server_port=port_to_use, )

