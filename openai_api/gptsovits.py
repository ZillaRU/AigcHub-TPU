from pydantic import BaseModel, Field
import os, io
from pydub import AudioSegment
from fastapi import Response
from api.base_api import BaseAPIRouter, change_dir, init_helper
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import soundfile as sf
import logging
from typing import Optional

app_name = "gptsovits"
logging.basicConfig(level=logging.INFO)

class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"
    @init_helper(dir)
    async def init_app(self):
        from repo.gptsovits.web_app import GptSovits_long, gptsovits_dir
        self.gptsovits_long = GptSovits_long(model_path=f"{gptsovits_dir}/models", tokenizer=f"{gptsovits_dir}/g2pw_tokenizer")
        return {"message": f"应用 {self.app_name} 已成功初始化。"}
    
    async def destroy_app(self):
        del self.gptsovits_long

router = AppInitializationRouter(app_name=app_name)


### 文本转语音 以及 音色克隆；兼容openai api，audio/speech
class TTSRequest(BaseModel):
    # 有意义的兼容参数
    input: str = Field(..., description="要转换为语音的文本")
    response_format: Optional[str] = Field('wav', description="音频格式")
    # 无意义的兼容参数
    model: Optional[str] = Field("gptsovits", description="（形式参数无意义）")
    voice: Optional[str] = Field('', description="（形式参数无意义）")
    speed: Optional[float] = Field(1.0, description="（形式参数无意义）")
    ## 专有参数
    audio_path: str = Field('', description="参考音色的路径")
    audio_content: str = Field('', description="参考音色的文本")


@router.post("/v1/audio/speech")
@change_dir(router.dir)

async def gptsovits(request: TTSRequest):  
    try:
        # 调用 gptsovits_long 函数
        sr, np_audio = router.gptsovits_long(request.audio_path, request.audio_content, request.input)

        wav_buffer = io.BytesIO()
        sf.write(file=wav_buffer, data=np_audio, samplerate=sr, format='WAV')
        buffer = wav_buffer
        response_format = request.response_format
        if response_format != 'wav':
            wav_audio = AudioSegment.from_wav(wav_buffer)
            wav_audio.frame_rate=sr
            buffer = io.BytesIO()
            wav_audio.export(buffer, format=response_format)

        return Response(content=buffer.getvalue(), media_type=f"audio/{response_format}")

    except Exception as e:
        return {"message": f"Error: {str(e)}"}