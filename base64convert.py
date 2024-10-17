import gradio as gr
import base64
from PIL import Image
import io
import numpy as np
import aiofiles

# 编码功能
def encode_text(text):
    try:
        encoded_bytes = base64.b64encode(text.encode())
        return encoded_bytes.decode()
    except Exception as e:
        return f"Error encoding text: {str(e)}"

def encode_image(image):
    try:
        buffered = io.BytesIO()
        pil_image = Image.fromarray(np.array(image))
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        return f"Error encoding image: {str(e)}"

async def encode_audio(file_path):
    try:
        async with aiofiles.open(file_path, "rb") as audio_file:
            audio_bytes = await audio_file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode()
        return encoded_audio
    except Exception as e:
        return f"Error encoding audio: {str(e)}"

# 解码功能
def decode_text(encoded_text):
    try:
        decoded_bytes = base64.b64decode(encoded_text.encode())
        return decoded_bytes.decode()
    except Exception as e:
        return f"Error decoding text: {str(e)}"

def decode_image(encoded_image):
    try:
        img_bytes = base64.b64decode(encoded_image.encode())
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except Exception as e:
        return f"Error decoding image: {str(e)}"

def decode_audio(encoded_audio):
    try:
        audio_bytes = base64.b64decode(encoded_audio.encode())
        return gr.Audio(audio_bytes, format='mp3')
    except Exception as e:
        return f"Error decoding audio: {str(e)}"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Base64 编解码器")

    with gr.Tabs() as tabs:
        with gr.TabItem("编码"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="输入文本", visible=True)
                    encode_text_button = gr.Button("编码文本")
                    image_input = gr.Image(label="上传图像", visible=True)
                    encode_image_button = gr.Button("编码图像")
                    audio_input = gr.Audio(sources="upload", type="filepath", label="上传音频", visible=True)
                    encode_audio_button = gr.Button("编码音频")
                with gr.Column():
                    encoded_output = gr.Textbox(label="Base64 编码", lines=10, interactive=True)

            encode_text_button.click(fn=encode_text, inputs=text_input, outputs=encoded_output)
            encode_image_button.click(fn=encode_image, inputs=image_input, outputs=encoded_output)
            encode_audio_button.click(fn=encode_audio, inputs=audio_input, outputs=encoded_output)

        with gr.TabItem("解码"):
            with gr.Row():
                with gr.Column():
                    decode_input = gr.Textbox(label="输入 Base64 编码文本/图像/音频", lines=10)
                with gr.Column():
                    decode_text_button = gr.Button("解码文本")
                    decoded_text_output = gr.Textbox(label="解码文本")
                    decode_image_button = gr.Button("解码图像")
                    decoded_image_output = gr.Image(label="解码图像")
                    decode_audio_button = gr.Button("解码音频")
                    decoded_audio_output = gr.Audio(label="解码音频")

            decode_text_button.click(fn=decode_text, inputs=decode_input, outputs=decoded_text_output)
            decode_image_button.click(fn=decode_image, inputs=decode_input, outputs=decoded_image_output)
            decode_audio_button.click(fn=decode_audio, inputs=decode_input, outputs=decoded_audio_output)

demo.launch(server_name='0.0.0.0', server_port=7860)
