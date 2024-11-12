import os
import gradio as gr
import numpy as np

DESCRIPTION = "Whisper 语音转文字"
STANDALONE = True

gr.Markdown(f"## {DESCRIPTION}",  elem_id="welcome_message")


if __name__ == "__main__":
    import argparse
    from repo.whisper.python.bmwhisper import load_model
    from repo.whisper.python.bmwhisper.transcribe import transcribe

    # 更改工作目录
    os.chdir("repo/whisper")

    def process_audio(audio, model_name, language, prompt, temperature):
        temperature_increment_on_fallback = 0.2
        if (increment := temperature_increment_on_fallback) is not None:
            temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
        else:
            temperature = [temperature]

        args = {
            'verbose': True,
            'task': 'transcribe',
            'language': None if language in ["", "string"] else language,
            'best_of': 5,
            'beam_size': 5,
            'patience': None,
            'length_penalty': None,
            'suppress_tokens': '-1',
            'initial_prompt': None if prompt in ["", "string"] else prompt,
            'condition_on_previous_text': True,
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            'word_timestamps': False,
            'prepend_punctuations': '"\'“¿([{-',
            'append_punctuations': '"\'.。,，!！?？:：”)]}、',
            'padding_size': 448
        }

        model = load_model({"model_name": model_name, "bmodel_dir": "models/BM1684X", "beam_size": 5, "padding_size": 448, "dev_id": 0})
        result = transcribe(model, audio, temperature=temperature, **args)

        return result["text"]

    iface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(sources="upload", type="filepath", label="音频文件"),
            gr.Dropdown(choices=["base"], value="base", label="模型名称"),
            gr.Dropdown(choices=["zh","en",None], value="base", label="语言"),
            gr.Textbox(value=None, label="初始提示"),
            gr.Number(value=0.0, label="温度"),
        ],
        outputs=gr.Textbox(label="转录结果"),
        title="Whisper语音转文字",
        description="上传音频文件并选择参数，将音频转录为文字。"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    iface.queue().launch(server_port=args.port, inbrowser=True)

