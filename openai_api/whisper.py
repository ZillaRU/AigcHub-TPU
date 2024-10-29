import time
import numpy as np
from subprocess import run, CalledProcessError
from api.base_api import BaseAPIRouter, change_dir, init_helper
from typing import Optional
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
import logging
logging.basicConfig(level=logging.INFO)


app_name = "whisper"


def load_audio(file: UploadFile, sr: int = 16000):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "pipe:1"
    ]
    try:
        out = run(cmd, input=file.file.read(), capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    finally:
        file.file.close()  # Ensure the file is closed after processing

    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        from repo.whisper.python.bmwhisper import load_model
        args = {}
        args["model_name"]    = "base"
        args["bmodel_dir"]    = "models/BM1684X"
        args["beam_size"]     = 5
        args["padding_size"]  = 448
        args["dev_id"]        = 0
        self.models = load_model(args)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.models

router = AppInitializationRouter(app_name=app_name)


### ASR；兼容openai api，audio/transcriptions
@router.post("/v1/audio/transcriptions")
@change_dir(router.dir)
async def whisper(
    file: UploadFile = File(...),
    model: Optional[str] = Form("base"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None)
):
    from repo.whisper.python.bmwhisper.transcribe import transcribe

    # init whisper parameters
    language = None if language in ["", "string"] else language
    prompt = None if prompt in ["", "string"] else prompt
    timestamp_granularities = None if timestamp_granularities in ["", "string"] else timestamp_granularities
    temperature_increment_on_fallback = 0.2
    if (increment := temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]
    logging.info(f"model: {model}, language: {language}, prompt: {prompt}, response_format: {response_format}, temperature: {temperature}, timestamp_granularities: {timestamp_granularities}")

    # Load the audio
    audio = load_audio(file)

    args = {'verbose': True, 'task': 'transcribe', 'language': language, 'best_of': 5, 'beam_size': 5, 'patience': None, 'length_penalty': None,
            'suppress_tokens': '-1', 'initial_prompt': prompt, 'condition_on_previous_text': True, 'compression_ratio_threshold': 2.4, 
            'logprob_threshold': -1.0, 'no_speech_threshold': 0.6, 'word_timestamps': False, 'prepend_punctuations': '"\'“¿([{-', 'append_punctuations': '"\'.。,，!！?？:：”)]}、', 'padding_size': 448}
    
    # Transcribe the audio
    logging.info("{:=^100}".format(f" Start "))
    audio_start_time = time.time()
    router.models.init_cnt()
    router.models.init_time()
    result = transcribe(router.models, audio, temperature=temperature, **args)
    total_time = time.time() - audio_start_time
    preprocess_time = total_time - router.models.inference_time
    router.models.print_cnt()
    logging.info(f"Preprocess time: {preprocess_time}s")
    logging.info(f"Inference time: {router.models.inference_time}s")
    logging.info(f"Total time: {total_time}s")

    # timestamp
    if timestamp_granularities is not None:
        transcription_data ={
            "task": "transcribe",
            "language": language,
            "text": result["text"],
            "segments": result["segments"],
        }
    else:
        transcription_data ={"text": result["text"]}
    # 响应格式
    if  response_format== "text":
        return PlainTextResponse(content=transcription_data["text"])
    else:
        return JSONResponse(content=transcription_data)
    

#### 测试命令
# curl http://localhost:8000/whisper/v1/audio/transcriptions \
#   -F 'file=@/data/AigcHub-TPU/repo/whisper/datasets/test/demo.wav;type=audio/wav' \
#   -F 'model=base'
