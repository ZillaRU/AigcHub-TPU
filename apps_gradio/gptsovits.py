import gradio as gr

DESCRIPTION = "音色克隆/TTS"
STANDALONE = True

gr.Markdown(f"## {DESCRIPTION}",  elem_id="welcome_message")

if __name__ == "__main__":
    import numpy as np
    import time
    import logging
    import os
    import argparse
    from repo.gptsovits.main import *
    from repo.gptsovits.utils import *

    os.chdir("repo/gptsovits")

    class GptSovits_long(GptSovits):
        def __init__(self, model_path='models', tokenizer='g2pw_tokenizer'):
            super().__init__(model_path, tokenizer)

        def prepare_input(self, text):
            text = text.strip("\n")
            if text[-1] not in splits:
                text += "。"
            norm_text = text_normalize(text)
            return norm_text

        def get_bert(self, norm_text, word2ph, len_ori_text):
            inputs = self.tokenizer(norm_text, return_tensors="np")
            a, b, c = inputs["input_ids"].astype(np.int32), inputs["token_type_ids"].astype(np.int32), inputs["attention_mask"].astype(np.int32)
            c[..., len_ori_text:] = 0

            res = self.bmodels([a, b, c], net_name='02_bert_35')[0]

            assert len(word2ph) == len(norm_text)
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = np.tile(res[i], (word2ph[i], 1))
                phone_level_feature.append(repeat_feature)
            phone_level_feature = np.concatenate(phone_level_feature)
            bert = phone_level_feature.T
            return bert

        def g2pw_bert_process(self, text, type='target'):
            norm_text = self.prepare_input(text)
            len_ori_text = len(norm_text)

            if type == 'target':
                num_dots = 2
            else:
                num_dots = 3
            how_many = sum(1 for x in norm_text if x in splits)
            delta = num_dots - how_many
            if delta > 0:
                norm_text = "." * delta + norm_text


            norm_text = fix_text_lenth(norm_text, 35)

            logging.info(f"文本处理：“{norm_text}”, 长度：{len(norm_text)}")

            phones1, word2ph1 = g2p(norm_text)  # g2pw模型
            phones1 = cleaned_text_to_sequence(phones1, "v2")
            id = find_penultimate_tag(phones1)
            phones1[id + 1:] = [2] * (len(phones1) - id - 1)

            bert1 = self.get_bert(norm_text, word2ph1, len_ori_text)  # BERT模型

            return bert1, phones1

        def __call__(self, ref_wav_path, ref_text, target_text, top_k=15, post_process=True, min_silence_len=200):
            try:
                start = time.time()
                ref_wav_16k = prepare_wav_input(ref_wav_path)
                ssl_content = self.bmodels([ref_wav_16k], net_name='00_cnhubert')[0]
                prompt = self.bmodels([ssl_content], net_name='01_vits_encoder')[0]

                bert2, phones2 = self.g2pw_bert_process(ref_text, type='ref')

                texts = process_long_text(target_text)
                audio_list = []
                zero_wav = np.zeros(int(self.hps.sampling_rate * 0.3), dtype=np.float32)

                for text in texts:
                    if text[-1] not in ['。', '！', '？']:
                        text = text[:-1] + '。'
                    bert1, phones1 = self.g2pw_bert_process(text)

                    bert = np.expand_dims(np.concatenate([bert2, bert1], 1), 0)
                    all_phoneme_ids = phones2 + phones1
                    all_phoneme_ids = np.expand_dims(np.int32(all_phoneme_ids), 0)

                    pred_semantic = self.t2s_process(all_phoneme_ids, bert, prompt, top_k)

                    SET_PRED_SEMANTIC_LEN = 200
                    if pred_semantic.shape[-1] < SET_PRED_SEMANTIC_LEN:
                        padding_size = SET_PRED_SEMANTIC_LEN - pred_semantic.shape[-1]
                        pred_semantic = np.pad(pred_semantic, pad_width=((0, 0), (0, padding_size)), mode='edge')
                    else:
                        pred_semantic = pred_semantic[:, :SET_PRED_SEMANTIC_LEN]

                    pred_semantic = np.expand_dims(pred_semantic, 0)
                    phones1 = np.expand_dims(np.int32(phones1), 0)
                    refer = get_spepc(self.hps, ref_wav_path)

                    audio_output = self.bmodels([pred_semantic, phones1, refer, self.randn_np], net_num=-1)[0][0, 0]

                    max_audio = np.abs(audio_output).max()
                    if max_audio > 1:
                        audio_output /= max_audio

                    audio_output = trim_audio_librosa(audio_output)
                    audio_output = np.concatenate([audio_output, zero_wav])

                    audio_output = (audio_output * 32768).astype(np.int16)
                    if post_process:
                        sr, audio_output = audio_post_process(self.hps.sampling_rate, audio_output, min_silence_len) 
                    audio_list.append(audio_output)

                audio_output = np.concatenate(audio_list)

                logging.info(f"耗时：{time.time() - start}")
                return self.hps.sampling_rate, audio_output
            except Exception as e:
                logging.error(f"处理音频时出错: {e}")
                raise

    logging.basicConfig(level=logging.INFO)
    gptsovits = GptSovits_long(gptsovits_dir + "/models", gptsovits_dir + "/g2pw_tokenizer")

    def process_audio(audio, ref_text, target_text, top_k, post_process=True, min_silence_len=200):
        sr, audio = gptsovits(audio, ref_text, target_text, top_k=top_k, post_process=post_process, min_silence_len=min_silence_len)
        return sr, audio

    iface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(sources="upload", type="filepath", label="参考语音(5~10s)"),
            gr.Textbox(lines=2, placeholder="输入参考文本...", label="参考文本(三个标点，35字以内。)"),
            gr.Textbox(lines=2, placeholder="输入目标文本...", label="目标文本(两句一切。)"),
            gr.Slider(value=15, label="Top-k", minimum=1, maximum=100, step=1),
            gr.Checkbox(value=False, label="是否进行后处理"),
            gr.Slider(value=200, label="静音检测阈值(毫秒)", minimum=100, maximum=500, step=50),
        ],
        outputs=gr.Audio(label="生成的语音"),
        title="GPT-SoVITS音色克隆",
        description="上传一段参考语音，并输入参考文本和目标文本，将生成一段新的语音(仅支持中文)。",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    iface.queue().launch(server_port=args.port,inbrowser=True)
