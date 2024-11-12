import gradio as gr

DESCRIPTION = "心流知镜语音对话"
STANDALONE = True

gr.Markdown(f"## {DESCRIPTION}",  elem_id="welcome_message")

if __name__ == "__main__":
    import numpy as np
    from repo.flowmirror.src.sail.modeling_flow_mirror_bmodel import *

    def layer_norm(x, epsilon=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / (std + epsilon)
        return normalized_x

    def app(question):
        # 预处理音频数据
        sr, wav = question
        if wav.ndim == 2:
            print("avarage")
            wav = wav.mean(0)
        assert wav.ndim == 1, wav.ndim
        if wav.shape[0] < 50000:
            wav = np.pad(wav, (0, 50000 - wav.shape[0]))
        else:
            wav = wav[:50000]
        wav = layer_norm(wav)
        wav = np.reshape(wav,(1,-1))

        # get input_ids by CNHubert
        input_ids = hubert.get_input_ids(wav)
        input_ids = np.pad(input_ids, ((0,0),(IDS_LENGTH - input_ids.shape[1], 0)))

        # Flowmirror Generation
        answer, text_completion = model.generate(prompt_input_ids=input_ids, speaker_embedding=speaker_embedding)
        
        answer = answer.squeeze()
        answer = answer - np.min(answer)
        max_audio=np.max(answer)
        answer/=max_audio
        answer = (answer * 32768).astype(np.int16)
        return 16000, answer
    
    # 加载模型
    hubert = CNHubert("models")
    model = FlowmirrorForConditionalGeneration(model_dir="models", config=Config("configs/config.json"), device_id=0)
    speaker_embedding = np.load("models/speaker_embedding.npz")['speaker_embedding_1']

    # 创建 Gradio 界面
    iface = gr.Interface(
        fn=app,
        inputs="audio",
        outputs="audio",
        title="心流知镜",
    )

    # 启动界面
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    iface.queue().launch(server_port=args.port,inbrowser=True)