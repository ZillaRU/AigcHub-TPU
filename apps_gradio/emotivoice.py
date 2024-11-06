import gradio as gr

DESCRIPTION = "Emotivoice 音色转换/TTS"
STANDALONE = True

gr.Markdown(f"## {DESCRIPTION}",  elem_id="welcome_message")



if __name__ == "__main__":
    import os
    import uuid
    import soundfile as sf
    from repo.emotivoice.demo_page import tts
    from repo.emotivoice.tone_color_conversion import get_se
    from repo.emotivoice.demo_page import get_models

    # 初始化模型
    os.chdir("repo/emotivoice")
    models, tone_color_converter, g2p, lexicon = get_models()

    def convert(src_wav, tgt_wav, tone_color_converter, get_se, save_path="/data/tmpdir/output.wav", encode_message=""):
        try:
            # 提取源说话人和目标说话人的音色特征
            source_se, _ = get_se(src_wav, tone_color_converter, target_dir='processed', vad=True)
            target_se, _  = get_se(tgt_wav, tone_color_converter, target_dir='processed', vad=True)
        except Exception as e:
            return {"error": f"Failed to extract speaker embedding: {e}"}
        tone_color_converter.convert(
            audio_src_path=src_wav, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        return save_path

    def text_to_speech(input_text, voice, emotion, audio_path):
        _name = f'/data/tmpdir/{str(uuid.uuid4())}.wav'
        src_wav = tts(input_text, emotion, voice, _name, models, g2p, lexicon)
        save_path = _name
        if audio_path and os.path.exists(audio_path):
            save_path = convert(src_wav=src_wav, tgt_wav=audio_path, tone_color_converter=tone_color_converter, get_se=get_se, encode_message='Airbox')
            if isinstance(save_path, dict):
                return {"text": save_path['error'], 'info': 'error message'}
        np_audio, sr = sf.read(save_path)
        return sr, np_audio

    def voice_changer(src_file, ref_file):
        save_path = convert(src_wav=src_file, tgt_wav=ref_file, tone_color_converter=tone_color_converter, get_se=get_se, encode_message='Airbox')
        if isinstance(save_path, dict):
            return {"text": save_path['error'], 'info': 'error message'}
        np_audio, sr = sf.read(save_path)
        return sr, np_audio

    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("## Emotivoice 音色转换与文本转语音")

        with gr.Tab("文本转语音"):
            with gr.Row():
                input_text = gr.Textbox(label="输入文本")
                voice = gr.Textbox(label="说话人ID", value="8051")
                emotion = gr.Textbox(label="情感提示", value="")
                audio_path = gr.Audio(label="目标音色路径", type="filepath")
                output_audio = gr.Audio(label="生成的语音")
            gr.Button("生成语音").click(text_to_speech, inputs=[input_text, voice, emotion, audio_path], outputs=output_audio)

        with gr.Tab("音色转换"):
            with gr.Row():
                src_file = gr.Audio(label="源音频文件", type="filepath")
                ref_file = gr.Audio(label="参考音频文件", type="filepath")
                output_audio = gr.Audio(label="转换后的音频")
            gr.Button("转换音色").click(voice_changer, inputs=[src_file, ref_file], outputs=output_audio)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.queue().launch(server_port=args.port, inbrowser=True, server_name="0.0.0.0")